#pragma once

#include "../DataChannel.hpp"

#include <mpi.h>
#include <memory>
#include <utility>
#include <unordered_map>
#include <vector>

namespace thd {

struct DataChannelMPI : DataChannel {
  struct RequestMPI : DataChannel::Request {
    friend class DataChannelMPI; // allows `DataChannelMPI` to access private members

    RequestMPI();
    virtual ~RequestMPI();

    virtual bool isCompleted() override;
    virtual void wait() override;

  private:
    template<typename T>
    void steal_buffer(std::shared_ptr<T> ptr);
    void steal_tensor_buffer(at::Tensor& t);
    MPI_Request& new_request();

    std::vector<std::shared_ptr<void>> _buffers;
    std::vector<at::Tensor> _tensor_buffers;
    std::vector<MPI_Request> _requests;
  };

  DataChannelMPI();
  virtual ~DataChannelMPI();

  bool init() override;

  rank_type getRank() override;
  rank_type getNumProcesses() override;

  void allGather(std::vector<at::Tensor>& output, at::Tensor& input,
                 THDGroup group_id = THDGroupWORLD) override;
  void gather(std::vector<at::Tensor>& output, at::Tensor& input,
              rank_type dst_rank, THDGroup group_id = THDGroupWORLD) override;
  void scatter(std::vector<at::Tensor>& input, at::Tensor& output,
               rank_type src_rank, THDGroup group_id = THDGroupWORLD) override;
  void allReduce(at::Tensor& data, THDReduceOp operation,
                 THDGroup group_id = THDGroupWORLD) override;
  void reduce(at::Tensor& data, THDReduceOp operation, rank_type dst_rank,
              THDGroup group_id = THDGroupWORLD) override;
  void broadcast(at::Tensor& data, rank_type src_rank,
                 THDGroup group_id = THDGroupWORLD) override;
  void send(Scalar& data, rank_type dst_rank) override;
  void send(at::Tensor& data, rank_type dst_rank) override;
  void receive(Scalar& data, rank_type src_rank) override;
  rank_type receive(at::Tensor& data) override;
  void receive(at::Tensor& data, rank_type src_rank) override;
  RequestMPI* isend(at::Tensor& data, rank_type dst_rank) override;
  RequestMPI* ireceive(at::Tensor& data, rank_type src_rank) override;

  void barrier(THDGroup group_id = THDGroupWORLD) override;
  THDGroup newGroup(const std::vector<rank_type>& ranks) override;

private:
  void _broadcastPack(at::Tensor& data, rank_type src_rank, MPI_Comm comm) const;
  void _broadcastUnpack(at::Tensor& data, rank_type src_rank, MPI_Comm comm) const;

  rank_type _rank; // Current process' rank
  rank_type _num_processes; // Number of processes in network

  // Existing groups of processes with assigned MPI communicator
  // and corresponding group ids
  std::unordered_map<THDGroup, std::pair<MPI_Comm, DataChannel::Group>> _groups;
};

} // namespace thd
