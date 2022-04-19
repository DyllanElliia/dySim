/*
 * @Author: DyllanElliia
 * @Date: 2022-04-14 17:35:20
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-19 17:14:04
 * @Description:
 */

#pragma once
#include "../render.hpp"

namespace dym {
namespace rt {
namespace {
std::vector<dym::Vector<Real, dym::PIC_RGB>> output;

// svgf data
std::vector<Vecor2> moment_history;
std::vector<Vecor3> color_history;
std::vector<Vecor2> moment_acc;
std::vector<Vecor3> color_acc;

std::vector<int> history_length;
std::vector<int> history_length_update;

std::vector<GBuffer> gbuffer_prev;

std::vector<Real> variance;

std::array<std::vector<Vector3>, 2> temp;

// A-Trous Filter
void ATrousFilter(Tensor<dym::Vector<Real, dym::PIC_RGB>>& colorin,
                  std::vector<dym::Vector<Real, dym::PIC_RGB>>& colorout,
                  std::vector<Real>& variance, Tensor<GBuffer>& gBuffer,
                  int level, bool is_last, Real sigma_c, Real sigma_n,
                  Real sigma_x, bool blur_variance, bool addcolor) {
  auto& res = colorin.shape();
  // 5x5 A-Trous kernel
  const Real h[25] = {
      1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0,
      1.0 / 64.0,  1.0 / 16.0, 3.0 / 32.0,  1.0 / 16.0, 1.0 / 64.0,
      3.0 / 128.0, 3.0 / 32.0, 9.0 / 64.0,  3.0 / 32.0, 3.0 / 128.0,
      1.0 / 64.0,  1.0 / 16.0, 3.0 / 32.0,  1.0 / 16.0, 1.0 / 64.0,
      1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0};

  // 3x3 Gaussian kernel
  const Real gaussian[9] = {1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
                            1.0 / 8.0,  1.0 / 4.0, 1.0 / 8.0,
                            1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0};

  const Vector2i g[9] = {
      Vector2i({-1, -1}), Vector2i({0, -1}), Vector2i({1, -1}),
      Vector2i({-1, 0}),  Vector2i({0, 0}),  Vector2i({1, 0}),
      Vector2i({-1, 1}),  Vector2i({0, 1}),  Vector2i({1, 1})};

  colorin.for_each_i([&](dym::Vector<Real, dym::PIC_RGB>& color, int x, int y) {
    int p = image.getIndexInt(gi(x, y));
    int step = 1 << level;

    Real var;
    // perform 3x3 gaussian blur on variance
    if (blur_variance) {
      Real sum = 0.0;
      Real sumw = 0.0;
      for (int sampleIdx = 0; sampleIdx < 9; sampleIdx++) {
        Vector2i loc = Vector2i({x, y}) + g[sampleIdx];
        if (loc.x >= 0 && loc.y >= 0 && loc.x < res[0] && loc.y < res[1]) {
          sum += gaussian[sampleIdx] * variance[colorin.getIndex(loc)];
          sumw += gaussian[sampleIdx];
        }
      }
      var = dym::max(sum / sumw, 0.0);
    } else {
      var = dym::max(variance[p], 0.0);
    }

    // Load pixel p data
    Real lp =
        0.2126 * colorin[p].x + 0.7152 * colorin[p].y + 0.0722 * colorin[p].z;
    Vector3 pp = gBuffer[p].position;
    Vector3 np = gBuffer[p].normal;

    Vector3 color_sum = Vector3(0.0);
    Real variance_sum = 0.0;
    Real weights_sum = 0;
    Real weights_squared_sum = 0;

    for (int i = -2; i <= 2; i++) {
      for (int j = -2; j <= 2; j++) {
        int xq = x + step * i;
        int yq = y + step * j;
        if (xq >= 0 && xq < res[0] && yq >= 0 && yq < res[1]) {
          int q = colorin.getIndexInt(gi(xq, yq));

          // Load pixel q data
          Real lq = 0.2126 * colorin[q][0] + 0.7152 * colorin[q][1] +
                    0.0722 * colorin[q].[2];
          Vector3& pq = gBuffer[q].position;
          Vector3& nq = gBuffer[q].normal;

          // Edge-stopping weights
          Real wl =
              dym::exp(-(lp - lq).length() / (dym::sqrt(var) * sigma_c + 1e-6));
          Real wn = dym::min(1.0, exp(-(np - nq).length() / (sigma_n + 1e-6)));
          Real wx = dym::min(1.0, exp(-(pp - pq).length() / (sigma_x + 1e-6)));

          // filter weights
          int k = (2 + i) + (2 + j) * 5;
          Real weight = h[k] * wl * wn * wx;
          weights_sum += weight;
          weights_squared_sum += weight * weight;
          color_sum += (colorin[q] * weight);
          variance_sum += (variance[q] * weight * weight);
        }
      }
    }

    // update color and variance
    if (weights_sum > 10e-6) {
      colorout[p] = color_sum / weights_sum;
      variance[p] = variance_sum / weights_squared_sum;
    } else {
      colorout[p] = colorin[p];
    }

    if (is_last && addcolor) {
      // colorout[p] *= gBuffer[p].albedo * gBuffer[p].ialbedo;
      colorout[p] *= gBuffer[p].albedo;
    }
  });
}

_DYM_FORCE_INLINE_ bool isReprjValid(Index<int>& res, Vector2i& curr_coord,
                                     Vector2i& prev_coord,
                                     Tensor<GBuffer>& curr_gbuffer,
                                     std::vector<GBuffer>& prev_gbuffer) {
  int p = curr_gbuffer.getIndexInt(curr_coord);
  int q = curr_gbuffer.getIndexInt(prev_coord);
  // reject if the pixel is outside the screen
  if (prev_coord.x() < 0 || prev_coord.x() >= res[0] || prev_coord.y() < 0 ||
      prev_coord.y() >= res[1])
    return false;
  // reject if the pixel is a different geometry
  if (prev_gbuffer[q].obj_id == -1 ||
      prev_gbuffer[q].obj_id != curr_gbuffer[p].obj_id)
    return false;
  // reject if the normal deviation is not acceptable
  if ((prev_gbuffer[q].normal - curr_gbuffer[p].normal).length_sqr() > 1e-2f)
    return false;
  return true;
}

// TODO: back projection
void BackProjection(
    std::vector<Real>& variacne_out, std::vector<int>& history_length,
    std::vector<int>& history_length_update,
    std::vector<Vector2>& moment_history, std::vector<Vector3>& color_history,
    std::vector<Vector2>& moment_acc, std::vector<Vector3>& color_acc,
    Tensor<dym::Vector<Real, dym::PIC_RGB>>& current_color,
    Tensor<GBuffer>& current_gbuffer, std::vector<GBuffer>& prev_gbuffer,
    Real color_alpha_min, Real moment_alpha_min) {
  const auto& res = current_color.shape();
  current_color.for_each_i([&](dym::Vector<Real, dym::PIC_RGB>& color, int x,
                               int y) {
    int p = image.getIndexInt(gi(x, y));
    int N = history_length[p];
    Vector3 sample = current_color[p];
    Real luminance =
        0.2126 * sample.x() + 0.7152 * sample.y() + 0.0722 * sample.z();

    if (N > 0 && current_gbuffer[p].obj_id != -1) {
      // Calculate NDC coordinates in previous frame (TODO: check correctness)
      auto& viewspace_position = current_gbuffer[p].position;
      Real clipx =
          viewspace_position.x() / viewspace_position.z() /** tanf(PI / 4)*/;
      Real clipy =
          viewspace_position.y() / viewspace_position.z() /** tanf(PI / 4)*/;
      Real ndcx = -clipx * 0.5 + 0.5;
      Real ndcy = -clipy * 0.5 + 0.5;
      Real prevx = ndcx * res[0] - 0.5;
      Real prevy = ndcy * res[1] - 0.5;
      /////////////

      bool v[4];
      Real floorx = floor(prevx);
      Real floory = floor(prevy);
      Real fracx = prevx - floorx;
      Real fracy = prevy - floory;

      bool valid =
          (floorx >= 0 && floory >= 0 && floorx < res[0] && floory < res[1]);

      // 2x2 tap bilinear filter
      Vector2i offset[4] = {Vector2i({0, 0}), Vector2i({1, 0}),
                            Vector2i({0, 1}), Vector2i({1, 1})};

      // check validity
      {
        for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++) {
          Vector2i loc =
              Vector2i({(int)floorx, (int)floory}) + offset[sampleIdx];
          v[sampleIdx] = isReprjValid(res, Vector2i({x, y}), loc,
                                      current_gbuffer, prev_gbuffer);
          valid = valid && v[sampleIdx];
        }
      }

      Vector3 prevColor = Vector3(0.0);
      Vector2 prevMoments = Vector2(0.0);
      Real prevHistoryLength = 0.0;

      if (valid) {
        // interpolate?
        Real sumw = 0.0;
        Real w[4] = {(1 - fracx) * (1 - fracy), fracx * (1 - fracy),
                     (1 - fracx) * fracy, fracx * fracy};

        for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++) {
          Vector2i loc =
              Vector2i({(int)floorx, (int)floory}) + offset[sampleIdx];
          int locq = current_color.getIndexInt(loc);
          if (v[sampleIdx]) {
            prevColor += w[sampleIdx] * color_history[locq];
            prevMoments += w[sampleIdx] * moment_history[locq];
            prevHistoryLength += w[sampleIdx] * (Real)history_length[locq];
            sumw += w[sampleIdx];
          }
        }
        if (sumw >= 0.01) {
          prevColor /= sumw;
          prevMoments /= sumw;
          prevHistoryLength /= sumw;
          // prevHistoryLength = 1;
          valid = true;
        }
      }

      // find suitable samples elsewhere
      if (!valid) {
        Real cnt = 0.0;
        const int radius = 1;

        for (int yy = -radius; yy <= radius; yy++) {
          for (int xx = -radius; xx <= radius; xx++) {
            Vector2 loc =
                Vector2({floorx, floory}) + Vector2({(Real)xx, (Real)yy});
            int q = current_color.getIndexInt(loc);
            if (isReprjValid(res, Vector2i({x, y}), loc, current_gbuffer,
                             prev_gbuffer)) {
              prevColor += color_history[q];
              prevMoments += moment_history[q];
              prevHistoryLength += history_length[q];
              cnt += 1.0;
            }
          }
        }

        if (cnt > 0.0) {
          prevColor /= cnt;
          prevMoments /= cnt;
          prevHistoryLength /= cnt;
          // prevHistoryLength = 0;
          valid = true;
        }
      }

      if (valid) {
        // calculate alpha values that controls fade
        Real color_alpha = dym::max(1.0 / (Real)(N + 1), color_alpha_min);
        Real moment_alpha = dym::max(1.0 / (Real)(N + 1), moment_alpha_min);

        // incresase history length
        history_length_update[p] = (int)prevHistoryLength + 1;

        // color accumulation
        color_acc[p] =
            current_color[p] * color_alpha + prevColor * (1.0 - color_alpha);

        // moment accumulation
        Real first_moment =
            moment_alpha * prevMoments.x() + (1.0 - moment_alpha) * luminance;
        Real second_moment = moment_alpha * prevMoments.y() +
                             (1.0 - moment_alpha) * luminance * luminance;
        moment_acc[p] = Vector2({first_moment, second_moment});

        // calculate variance from moments
        Real variance = second_moment - first_moment * first_moment;
        variacne_out[p] = variance > 0.0 ? variance : 0.0;
        return;
      }
    }

    // If there's no history
    history_length_update[p] = 1;
    color_acc[p] = current_color[p];
    moment_acc[p] = Vector2({luminance, luminance * luminance});
    variacne_out[p] = 100.0;
  });
}

// Estimate variance spatially
__global__ void EstimateVariance(
    Tensor<dym::Vector<dym::Pixel, dym::PIC_RGB>>& image,
    std::vector<Real>& variacne, std::vector<Vector3>& color) {
  image.for_each_i([&](dym::Vector<Real, dym::PIC_RGB>& color, int x, int y) {
    int p = image.getIndexInt(gi(x, y));
    // TODO
    variacne[p] = 10.0;
  });
}

}  // namespace

void init_svgf(Index<int> shape) {
  const int pixelcount = shape[0] * shape[1];
  output.resize(pixelcount);
  moment_history.resize(pixelcount);
  color_history.resize(pixelcount);
  moment_acc.resize(pixelcount);
  color_acc.resize(pixelcount);

  history_length.resize(pixelcount);
  history_length_update.resize(pixelcount);

  gbuffer_prev.resize(pixelcount);

  variance.resize(pixelcount);

  temp[0].resize(pixelcount);
  temp[1].resize(pixelcount);
}

void denoise_svgf(Tensor<dym::Vector<Real, dym::PIC_RGB>>& input,
                  Tensor<GBuffer>& gbuffer) {}
}  // namespace rt
}  // namespace dym