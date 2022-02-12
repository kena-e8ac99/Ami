#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <utility>

#include "ami/concepts/activation_function.hpp"
#include "ami/layer/component/bias.hpp"
#include "ami/layer/component/node.hpp"

namespace ami {

  template <std::size_t OutputSize, activation_function Func>
  requires (OutputSize > 0)
  struct dense_layer final {
    template <std::floating_point RealType, std::size_t InputSize>
    requires (InputSize > 0)
    class type final {
    public:
      // Public Types
      using size_type  = std::size_t;
      using real_type  = RealType;
      using node_type  = node<RealType, InputSize>;
      using bias_type  = bias<RealType>;
      using value_type = std::pair<
          std::array<typename node_type::value_type, OutputSize>,
          std::array<real_type, OutputSize>>;

      // Public Static Members
      static constexpr size_type input_size  = InputSize;
      static constexpr size_type output_size = OutputSize;

      // Constructor
      type() = default;

      explicit constexpr type(const value_type& value)
          : nodes_{[&value = value.first]<std::size_t... I>(
                std::index_sequence<I...>) {
              return std::array<node_type, output_size>{
                  (static_cast<void>(I), node_type{(value.first)[I]})...};
            }(std::make_index_sequence<output_size>{})},
            bias_{[&value = value.second]<std::size_t... I>(
                std::index_sequence<I...>) {
              return std::array<bias_type, output_size>{
                  (static_cast<void>(I), bias_type{(value.second)[I]})...};
            }(std::make_index_sequence<output_size>{})}
      {}

      explicit constexpr type(value_type&& value)
          : nodes_{[value = std::move(value.first)]<std::size_t... I>(
                std::index_sequence<I...>) {
              return std::array<node_type, output_size>{
                  (static_cast<void>(I), node_type{std::move(value[I])})...};
            }(std::make_index_sequence<output_size>{})},
            bias_{[value = std::move(value.second)]<std::size_t... I>(
                std::index_sequence<I...>) {
              return std::array<bias_type, output_size>{
                  (static_cast<void>(I), bias_type{std::move(value[I])})...};
            }(std::make_index_sequence<output_size>{})}
      {}

      // Getter
      constexpr auto value() const& noexcept {
        auto first = [&]<std::size_t... I>(std::index_sequence<I...>) {
          return typename value_type::first_type{nodes_[I].value()...};
        }(std::make_index_sequence<output_size>{});

        auto second = [&]<std::size_t... I>(std::index_sequence<I...>) {
          return typename value_type::second_type{bias_[I].value()...};
        }(std::make_index_sequence<output_size>{});

        return std::pair{std::move(first), std::move(second)};
      }

      constexpr auto value() && noexcept {
        auto first = [&]<std::size_t... I>(std::index_sequence<I...>) {
          return typename value_type::first_type{
              std::move(nodes_[I]).value()...};
        }(std::make_index_sequence<output_size>{});

        auto second = [&]<std::size_t... I>(std::index_sequence<I...>) {
          return typename value_type::second_type{
              std::move(bias_[I]).value()...};
        }(std::make_index_sequence<output_size>{});

        return std::pair{std::move(first), std::move(second)};
      }

    private:
      // Private Members
      std::array<node_type, output_size> nodes_{};
      std::array<bias_type, output_size> bias_{};
    };
  };

  template <std::floating_point RealType, std::size_t InputSize,
            std::size_t OutputSize, activation_function Func>
  using dense_layer_t =
      typename  dense_layer<OutputSize, Func>::template
          type<RealType, InputSize>;
}
