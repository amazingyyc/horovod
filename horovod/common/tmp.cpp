
/**
 * multi-thread for hierarchical allreduce for nccl
 * every thread have a unique nccl comm but they are share the MPI comm
 */
void hierarchical_allreduce_nccl(HorovodGlobalState& global_state, ncclComm_t nccl_comm, MPI_Comm cross_comm, std::vector<TensorTableEntry> entries) {
	assert(!entries.empty());

	auto& firest_entry = entries[0];

	/**must on gpu*/
	assert(CPU_DEVICE_ID != firest_entry.device);

	int size = global_state.size;
	int local_size = global_state.local_size;
	int cross_size = global_state.cross_size;

	int rank = global_state.rank;
	int local_rank = global_state.local_rank;
	int cross_rank = global_state.cross_rank;

	/**create a stream for every thread*/
	cudaStream_t cuda_stream;

	int greatest_priority;
	CUDA_CHECK(entries, "cudaSetDevice", cudaSetDevice(first_entry.device));
	CUDA_CHECK(entries, "cudaDeviceGetStreamPriorityRange", cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
	CUDA_CHECK(entries, "cudaStreamCreateWithPriority", cudaStreamCreateWithPriority(&cuda_stream, cudaStreamNonBlocking, greatest_priority));

	/**malloc a GPU to fusion all the entry data*/
	int64_t num_elements = 0;
	int64_t num_bytes = 0;

	int element_size;
	MPI_Type_size(GetMPIDataType(first_entry.tensor), &element_size);

	for (auto& item : entries) {
		num_elements += item.tensor->shape().num_elements();
		num_bytes += item.tensor->size();
	}

	/**
	 * malloc fusion buffer
	 * make the fusion_buffer can be divisible by local_size and FUSION_BUFFER_ATOMIC_UNIT
	 * than when split to reduce scatter it can be improve performance
	 * the FUSION_BUFFER_ATOMIC_UNIT is 64 so it always can be divisible by element_size
	 */
	int64_t div = local_size * FUSION_BUFFER_ATOMIC_UNIT;
	int64_t fusion_buffer_size = ((num_bytes + div - 1) / div) * div;

	/**make sure the fusion_buffer_size can be divisible by element_size*/
	assert(0 == fusion_buffer_size % element_size && fusion_buffer_size > 0);

	void* fusion_buffer = nullptr;

	CUDA_CHECK(entries, "cudaMalloc", cudaMalloc(&fusion_buffer, (size_t)fusion_buffer_size));

	/**copy memory to fusion_buffer*/
	int64_t offset = 0;
	for (auto& item : entries) {
		void* fusion_buffer_offset = (uint8_t*)fusion_buffer + offset;

		CUDA_CHECK(entries, "cudaMemcpyAsync", cudaMemcpyAsync(fusion_buffer_offset, item.tensor->data(), (size_t)item.tensor->size(), cudaMemcpyDeviceToDevice, cuda_stream));

		offset += item.tensor->size();
	}

	int64_t fusion_num_elements = fusion_buffer_size / element_size;

	/**make sure the fusion_num_elements can be divisible by local_size*/
	assert(0 == fusion_num_elements % local_size && fusion_num_elements > 0);

	/**the element num in per rank and buffer len, the last rank may have some dummy data*/
	int64_t num_elements_per_rank = fusion_num_elements / local_size;
	int64_t buffer_len_per_rank = element_size * num_elements_per_rank;
	void* fusion_buffer_per_rank_offset = (uint8_t*)fusion_buffer + buffer_len_per_rank * local_rank;

	/**start to reduceSctter*/
	NCCL_CHECK(entries, "ncclReduceScatter", ncclReduceScatter(
		fusion_buffer,
		fusion_buffer_per_rank_offset,
		(size_t)num_elements_per_rank,
		GetNCCLDataType(first_entry.tensor),
		ncclSum,
		nccl_comm,
		cuda_stream));

	/**
	 * for now the fusion_buffer have contain the reduceScatter'data
	 * now should copy the GPU memory to CPU
	 */
	void* host_buffer = malloc(buffer_len_per_rank);

	CUDA_CHECK(entries, "cudaMemcpyAsync", cudaMemcpyAsync(host_buffer,
		fusion_buffer_per_rank_offset,
		buffer_len_per_rank,
		cudaMemcpyDeviceToHost,
		cuda_stream));

	/**
	 * for now the data have been copy to host_buffer, now it need use the MPI to do all reduce cross machine
	 */
	MPI_CHECK(entries, "MPI_Allreduce", MPI_Allreduce(MPI_IN_PLACE,
		host_buffer,
		(int)num_elements_per_rank,
		GetMPIDataType(first_entry.tensor),
		first_entry.tensor->dtype() == HOROVOD_FLOAT16 ? horovod_global.mpi_float16_sum : MPI_SUM,
		cross_comm));

	/**copy data to GPU*/
	CUDA_CHECK(entries, "cudaMemcpyAsync", cudaMemcpyAsync(fusion_buffer_per_rank_offset,
		host_buffer,
		buffer_len_per_rank, cudaMemcpyHostToDevice,
		cuda_stream));

	NCCL_CHECK(entries, "ncclAllGather", ncclAllGather(fusion_buffer_per_rank_offset,
		fusion_buffer,
		(size_t)num_elements_per_rank,
		GetNCCLDataType(first_entry.tensor),
		nccl_comm,
		cuda_stream));

	/**copy buffer back*/
	offset = 0;

	for (auto& item : entries) {
		void* fusion_buffer_offset = (uint8_t*)fusion_buffer + offset;

		CUDA_CHECK(entries, "cudaMemcpyAsync", cudaMemcpyAsync((void*)item.output->data(),
			fusion_buffer_offset,
			(size_t)item.tensor->size(),
			cudaMemcpyDeviceToDevice,
			cuda_stream));
		offset += item.tensor->size();
	}

	/**free memory*/
	free(host_buffer);
	CUDA_CHECK(entries, "cudaFree", cudaFree(fusion_buffer);

	/**destroy comm*/
	NCCL_CHECK(entries, "ncclCommDestroy", ncclCommDestroy(nccl_comm));

	MPI_Comm_free(&cross_comm);
}