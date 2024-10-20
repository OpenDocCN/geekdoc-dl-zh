<!--yml

category: 未分类

date: 2024-07-01 18:18:28

-->

# 内核中的巧妙宏技巧：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/02/kernel-macro-tricks/`](http://blog.ezyang.com/2010/02/kernel-macro-tricks/)

## 内核中的巧妙宏技巧

给 C 程序员的一个经典风格提示是，尽可能使用内联函数而不是宏。这个建议源于宏和内联函数可以达到相同的效果，但内联函数还可以进行类型检查。

结果表明，如果愿意采用 Linux 内核下面这个小巧的技巧，你*确实可以*通过宏实现静态类型检查：

```
#define module_param_named(name, value, type, perm) \
        param_check_##type(name, &(value)); \
        module_param_call(name, param_set_##type, param_get_##type, &value, perm); \
        __MODULE_PARM_TYPE(name, #type)

```

嗯... 我想知道那个`param_check_##type`调用是怎么回事。再深入挖掘几个宏定义，我们看到：

```
#define __param_check(name, p, type) \
        static inline type *__check_##name(void) { return(p); }

```

就是这样。一个名为`__check_##name`的一次性内联函数确保`p`与`type`是相同类型。还附有一条注释，解释了发生了什么：

```
/* The macros to do compile-time type checking stolen from Jakub
   Jelinek, who IIRC came up with this idea for the 2.4 module init code. */

```
