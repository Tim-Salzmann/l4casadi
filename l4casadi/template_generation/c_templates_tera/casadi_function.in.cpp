#include <l4casadi.hpp>
#include <iostream>

L4CasADi l4casadi("{{ model_path }}", "{{ name }}");

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


static const casadi_int casadi_s_in0[3] = { {{ rows_in }}, {{ cols_in }}, 1};
static const casadi_int casadi_s_out0[3] = { {{ rows_out }}, {{ cols_out }}, 1};


CASADI_SYMBOL_EXPORT int {{ name }}(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  l4casadi.forward(arg[0], {{ rows_in }}, {{ cols_in }}, res[0]);
  return 0;
}

CASADI_SYMBOL_EXPORT int jac_{{ name }}(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  l4casadi.jac(arg[0], {{ rows_in }}, {{ cols_in }}, res[0]);
  return 0;
}

/*
CASADI_SYMBOL_EXPORT int {{ name }}_alloc_mem(void) { return 0; }

CASADI_SYMBOL_EXPORT int {{ name }}_init_mem(int mem) { return 0; }

CASADI_SYMBOL_EXPORT void {{ name }}_free_mem(int mem) {}

CASADI_SYMBOL_EXPORT int {{ name }}_checkout(void) { return 0; }

CASADI_SYMBOL_EXPORT void {{ name }}_release(int mem) {}

CASADI_SYMBOL_EXPORT void {{ name }}_incref(void) {}

CASADI_SYMBOL_EXPORT void {{ name }}_decref(void) {}
*/

// Only single input, single output is supported at the moment
CASADI_SYMBOL_EXPORT casadi_int {{ name }}_n_in(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_int {{ name }}_n_out(void) { return 1;}

/*
CASADI_SYMBOL_EXPORT casadi_real {{ name }}_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* {{ name }}_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* {{ name }}_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}
*/

CASADI_SYMBOL_EXPORT const casadi_int* {{ name }}_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s_in0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* {{ name }}_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s_out0;
    default: return 0;
  }
}

/*
CASADI_SYMBOL_EXPORT int {{ name }}_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 0;
  if (sz_res) *sz_res = 0;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}
*/


#ifdef __cplusplus
} /* extern "C" */
#endif