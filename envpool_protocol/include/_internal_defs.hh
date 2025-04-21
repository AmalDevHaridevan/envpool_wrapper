#include "definitions.hh"
// #ifndef CLSNAME
// #error "CLSNAME not defined"
// #endif
// #define EXPAND(arg) arg
// #define EXPAND2(arg) EXPAND(EXPAND(arg))

// #define CONCAT(a, b) a##b
// #define CONCAT_EXPAND(a, b) CONCAT(a, b)
#define STR(x) #x
#define XSTR(x) STR(x)

// // #define ENVFNS_CLASS_GEN(name) name ## Fns
// #define ENVFNCLS CONCAT_EXPAND(CLSNAME, Fns)
// // #define SPECCLS_CLASS_GEN(name) name ## Spec
// #define SPECCLS CONCAT_EXPAND(CLSNAME, Spec)
// // #define POOLCLS_CLASS_GEN(name)  name ## Pool
// #define POOLCLS CONCAT_EXPAND(CLSNAME, Pool)


// // // #define GEN_PY_TYPES(name) name ## PY
// // #define PY_SPEC  CONCAT_EXPAND(CLSNAME, SpecPY)
// // #define PY_POOL  CONCAT_EXPAND(CLSNAME, PoolPY)
// // #define GEN_PY_MODULE_NAME(name) name ## _py
// #define PYMODULE_NAME CLSNAME