<!--yml

分类：未分类

日期：2024-07-01 18:17:18

-->

# HoTT 在 Coq 中的练习（正在进行中）：ezyang 的博客

> 来源：[`blog.ezyang.com/2013/07/hott-exercises-in-coq-in-progress/`](http://blog.ezyang.com/2013/07/hott-exercises-in-coq-in-progress/)

昨天在飞机上，我花了些时间用 Coq 实现了《HoTT 书》中的练习。我完成了 1.6 部分（是的，进展不大，也许我应该建一个 GitHub 仓库，如果其他人有兴趣贡献框架的话。不过，对于解决方案我还不知道该怎么办）。所有这些都已经测试求解了。

为了运行这个开发，你需要 HoTT/coq；安装说明在[这里。](https://github.com/HoTT/HoTT/blob/master/INSTALL.txt)

**更新。** 可以在[HoTT-coqex](https://github.com/ezyang/HoTT-coqex)存储库中找到解决方案和更多练习。我完成了所有非平凡的同伦特定练习，并略过了一些更标准的类型理论练习（这些并不真正是同伦特定的）。有些解决方案实在太糟糕了，需要一些改进。

```
Require Import HoTT.

Definition admit {T: Type} : T. Admitted.

(* Exercise 1.1 *)
Definition mycompose {A B C : Type} (g : B -> C) (f : A -> B) : A -> C := admit.

Goal forall (A B C D : Type) (f : A -> B) (g : B -> C) (h : C -> D),
       mycompose h (mycompose g f) = mycompose (mycompose h g) f.
Admitted.

(* Exercise 1.2 *)
Section ex_1_2_prod.
  Variable A B : Type.
  Check @fst.
  Check @snd.
  Definition my_prod_rec (C : Type) (g : A -> B -> C) (p : A * B) : C := admit.
  Goal fst = my_prod_rec A (fun a => fun b => a). Admitted.
  Goal snd = my_prod_rec B (fun a => fun b => b). Admitted.
End ex_1_2_prod.

Section ex_1_2_sig.
  Variable A : Type.
  Variable B : A -> Type.
  Check @projT1.
  Check @projT2.
  Definition my_sig_rec (C : Type) (g : forall (x : A), B x -> C) (p : exists (x : A), B x) : C := admit.
  Goal @projT1 A B = my_sig_rec A (fun a => fun b => a). Admitted.
  (* What goes wrong when you try to prove this for projT2? *)
End ex_1_2_sig.

(* Exercise 1.3 *)

Definition refl {A : Type} (x : A) : x = x := 1%path.

Section ex_1_3_prod.
  Variable A B : Type.
  (* Given by the book *)
  Definition uppt : forall (x : A * B), ((fst x, snd x) = x) :=
    fun p => match p with (a,b) => refl (a,b) end.
  Definition my_prod_ind (C : A * B -> Type) (g : forall (x : A) (y : B), C (x, y)) (x : A * B) : C x := admit.
  Goal forall C g a b, my_prod_ind C g (a, b) = g a b. Admitted.
End ex_1_3_prod.

Section ex_1_3_sig.
  Variable A : Type.
  Variable B : A -> Type.
  Definition sig_uppt : forall (x : exists (a : A), B a), ((projT1 x; projT2 x) = x) := admit.
  Definition mysig_ind (C : (exists (a : A), B a) -> Type) (g : forall (a : A) (b : B a), C (a; b)) (x : exists (a : A), B a) : C x := admit.
  Goal forall C g a b, mysig_ind C g (a; b) = g a b. Admitted.
End ex_1_3_sig.

(* Exercise 1.4 *)
Fixpoint iter (C : Type) (c0 : C) (cs : C -> C) (n : nat) : C :=
  match n with
    | 0 => c0
    | S n' => cs (iter C c0 cs n')
  end.
Definition mynat_rec (C : Type) : C -> (nat -> C -> C) -> nat -> C := admit.
Eval compute in mynat_rec (list nat) nil (@cons nat) 2.
Eval compute in nat_rect (fun _ => list nat) nil (@cons nat) 2.

(* Exercise 1.5 *)
Definition mycoprod (A B : Type) := exists (x : Bool), Bool_rect (fun _ => Type) A B x.

Section ex_1_5.
  Variable A B : Type.
  Definition inl := existT (Bool_rect (fun _ => Type) A B) true.
  Definition inr := existT (Bool_rect (fun _ => Type) A B) false.
  Definition mycoprod_ind (C : mycoprod A B -> Type)
                          (l : forall (a : A), C (inl a))
                          (r : forall (b : B), C (inr b))
                          (x : mycoprod A B) : C x := admit.
  Goal forall C l r x, mycoprod_ind C l r (inl x) = l x. Admitted.
  Goal forall C l r x, mycoprod_ind C l r (inr x) = r x. Admitted.
End ex_1_5.

(* Exercise 1.6 *)

Definition myprod (A B : Type) := forall (x : Bool), Bool_rect (fun _ => Type) A B x.
Section ex_1_6.
  Context `{Funext}.
  Variable A B : Type.
  Definition mypr1 (p : myprod A B) := p true.
  Definition mypr2 (p : myprod A B) := p false.
  Definition mymkprod (a : A) (b : B) : myprod A B := Bool_rect (Bool_rect (fun _ => Type) A B) a b.
  Definition myprod_ind (C : myprod A B -> Type)
                        (g : forall (x : A) (y : B), C (mymkprod x y)) (x : myprod A B) : C x := admit.
  Goal forall C g a b, myprod_ind C g (mymkprod a b) = g a b. Admitted.
End ex_1_6.

```

实际上，我撒了个谎。我还没有证明练习 1.6 中的最后一个目标；我的困难在于我不知道如何让函数外延性计算，但我确信这是一些简单的事情...
