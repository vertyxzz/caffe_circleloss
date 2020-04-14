/*
name: CircleLossLayer
file: circle_loss_layer.hpp/cpp/cu
date        author  log
------------------------------------
2020-04-13  wujp    created
*/

#ifndef CAFFE_CIRCLE_LOSS_LAYER_HPP_
#define CAFFE_CIRCLE_LOSS_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
class CircleLossLayer : public LossLayer<Ftype, Btype> {
  typedef Ftype Dtype;
 public:
  explicit CircleLossLayer(const LayerParameter& param)
      : LossLayer<Ftype, Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "CircleLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
    
 protected:
  void ComputeDiff_cpu(const Dtype* x_1, const Dtype* x_2,
      const Dtype x_1_norm, const Dtype x_2_norm, const Dtype inner_val, Dtype* x_1_diff);
  void ComputeDiff_gpu(const Dtype* x_1, const Dtype* x_2,
      const Dtype x_1_norm, const Dtype x_2_norm, const Dtype inner_val, Dtype* x_1_diff);
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
      
  Dtype calc_softplus(Dtype x);
  Dtype calc_logsumexp(const Dtype* x, int x_len, Dtype* prob = nullptr);
  
  TBlob<Dtype> inner_matrix_;
  TBlob<Dtype> norm_;
  TBlob<Dtype> bottom_diff_;
  TBlob<Dtype> diff_1_;
  TBlob<Dtype> diff_2_;  
  
  TBlob<Dtype> logit_n_;
  TBlob<Dtype> logit_p_;
  TBlob<Dtype> prob_n_;
  TBlob<Dtype> prob_p_;
  TBlob<Dtype> cos_n_;
  TBlob<Dtype> cos_p_;
  TBlob<Dtype> inner_n_;
  TBlob<Dtype> inner_p_;
  TBlob<Dtype> idx_n1_;
  TBlob<Dtype> idx_n2_;
  TBlob<Dtype> idx_p1_;
  TBlob<Dtype> idx_p2_;  
  
  int batch_size_;
  int fea_dim_;
  Dtype gamma_;
  Dtype margin_;
  Dtype delta_p_;
  Dtype delta_n_;
  Dtype optimum_p_;
  Dtype optimum_n_;  
};

}

#endif
