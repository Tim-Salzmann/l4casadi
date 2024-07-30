#include <l4casadi.hpp>

L4CasADi l4casadi("{{ model_path }}", "{{ name }}", {{ rows_in }}, {{ cols_in }}, {{ rows_out }}, {{ cols_out }}, "{{ device }}", {{ has_jac }}, {{ has_adj1 }}, {{ has_jac_adj1 }}, {{ has_jac_jac }}, {{ scripting }}, {{ model_is_mutable }});

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

// Function {{ name }}

static const casadi_int {{ name }}_s_in0[3] = { {{ rows_in }}, {{ cols_in }}, 1};
static const casadi_int {{ name }}_s_out0[3] = { {{ rows_out }}, {{ cols_out }}, 1};

// Only single input, single output is supported at the moment
CASADI_SYMBOL_EXPORT casadi_int {{ name }}_n_in(void) { return 1;}
CASADI_SYMBOL_EXPORT casadi_int {{ name }}_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT const casadi_int* {{ name }}_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return {{ name }}_s_in0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* {{ name }}_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return {{ name }}_s_out0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int {{ name }}(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  l4casadi.forward(arg[0], res[0]);
  return 0;
}

{% if has_jac == "true" %}
// Jacobian {{ name }}

CASADI_SYMBOL_EXPORT casadi_int jac_{{ name }}_n_in(void) { return 2;}
CASADI_SYMBOL_EXPORT casadi_int jac_{{ name }}_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT int jac_{{ name }}(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  l4casadi.jac(arg[0], res[0]);
  return 0;
}

{% if batched == "true" %}
// Sparse output if batched.
static const casadi_int jac_{{ name }}_s_out0[{{jac_ccs_len}}] = { {{ jac_ccs }}};

CASADI_SYMBOL_EXPORT const casadi_int* jac_{{ name }}_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return jac_{{ name }}_s_out0;
    default: return 0;
  }
}
{% endif %}
{% endif %}


{% if has_adj1 == "true" %}
// adj1 {{ name }}

CASADI_SYMBOL_EXPORT casadi_int adj1_{{ name }}_n_in(void) { return 3;}
CASADI_SYMBOL_EXPORT casadi_int adj1_{{ name }}_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT int adj1_{{ name }}(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  // adj1 [i0, out_o0, adj_o0] -> [out_adj_i0]
  l4casadi.adj1(arg[0], arg[2], res[0]);
  return 0;
}
{% endif %}


{% if has_jac_adj1 == "true" %}
// jac_adj1 {{ name }}

CASADI_SYMBOL_EXPORT casadi_int jac_adj1_{{ name }}_n_in(void) { return 4;}
CASADI_SYMBOL_EXPORT casadi_int jac_adj1_{{ name }}_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT int jac_adj1_{{ name }}(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  // jac_adj1 [i0, out_o0, adj_o0, out_adj_i0] -> [jac_adj_i0_i0, jac_adj_i0_out_o0, jac_adj_i0_adj_o0]
  if (res[1] != NULL) {
    l4casadi.invalid_argument("jac_adj_i0_out_o0 is not provided by L4CasADi. If you need this feature, please contact the L4CasADi developer.");
  }
  if (res[2] != NULL) {
    l4casadi.invalid_argument("jac_adj_i0_adj_o0 is not provided by L4CasADi. If you need this feature, please contact the L4CasADi developer.");
  }
  if (res[0] == NULL) {
    l4casadi.invalid_argument("L4CasADi can only provide jac_adj_i0_i0 for jac_adj1_{{ name }} function. If you need this feature, please contact the L4CasADi developer.");
  }
  l4casadi.jac_adj1(arg[0], arg[2], res[0]);
  return 0;
}

{% if batched == "true" %}
// Sparse output if batched.
static const casadi_int jac_adj1_{{ name }}_s_out0[{{jac_adj_ccs_len}}] = { {{ jac_adj_ccs }}};
static const casadi_int jac_adj1_{{ name }}_s_out23[3] = { {{ rows_in }} * {{ cols_in }}, {{ rows_out }} * {{ cols_out }}, 1};

CASADI_SYMBOL_EXPORT const casadi_int* jac_adj1_{{ name }}_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return jac_adj1_{{ name }}_s_out0;
    case 1: return jac_adj1_{{ name }}_s_out23;
    case 2: return jac_adj1_{{ name }}_s_out23;
    default: return 0;
  }
}
{% endif %}
{% endif %}


{% if has_jac_jac == "true" %}
// jac_jac {{ name }}

CASADI_SYMBOL_EXPORT int jac_jac_{{ name }}(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
   // [i0, out_o0, out_jac_o0_i0] -> [jac_jac_o0_i0_i0, jac_jac_o0_i0_out_o0]
   if (res[1] != NULL) {
    l4casadi.invalid_argument("jac_jac_o0_i0_out_o0 is not provided by L4CasADi. If you need this feature, please contact the L4CasADi developer.");
  }
  if (res[0] == NULL) {
    l4casadi.invalid_argument("L4CasADi can only provide jac_jac_o0_i0_i0 for jac_jac_{{ name }} function. If you need this feature, please contact the L4CasADi developer.");
  }
  l4casadi.jac_jac(arg[0], res[0]);
  return 0;
}

{% if batched == "true" %}
// jac_jac {{ name }}

static const casadi_int jac_jac_{{ name }}_s_out0[{{jac_jac_ccs_len}}] = { {{ jac_jac_ccs }}};
static const casadi_int jac_jac_{{ name }}_s_out1[3] = { {{ rows_in }} * {{ cols_in }} * {{ rows_out }} * {{ cols_out }}, {{ rows_out }} * {{ cols_out }}, 1};
CASADI_SYMBOL_EXPORT const casadi_int* jac_jac_{{ name }}_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return jac_jac_{{ name }}_s_out0;
    case 1: return jac_jac_{{ name }}_s_out1;
    default: return 0;
  }
}
CASADI_SYMBOL_EXPORT casadi_int jac_jac_{{ name }}_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int jac_jac_{{ name }}_n_out(void) { return 2;}
{% endif %}
{% endif %}

#ifdef __cplusplus
} /* extern "C" */
#endif
