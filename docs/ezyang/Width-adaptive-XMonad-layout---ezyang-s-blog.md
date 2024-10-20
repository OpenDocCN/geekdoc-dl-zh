<!--yml

category: 未分类

date: 2024-07-01 18:17:11

-->

# 宽度自适应的 XMonad 布局 : ezyang’s 博客

> 来源：[`blog.ezyang.com/2015/05/width-adaptive-xmonad-layout/`](http://blog.ezyang.com/2015/05/width-adaptive-xmonad-layout/)

## 宽度自适应的 XMonad 布局

我通常的笔记本设置是使用宽屏显示器，并将笔记本屏幕作为辅助显示器。长期以来，我使用了两种 XMonad 布局：一种是针对笔记本显示器的全屏布局（我使用大字体以便于眼睛放松），另一种是在大屏幕上使用的两列布局。

但我遇到了一个恼人的问题：如果我从小屏幕切换工作区到大屏幕，XMonad 仍然会使用全屏布局，我必须通过 Alt-Tab 切换到两列布局。更让人气愤的是，如果我又切回去，我还得再次 Alt-Tab。

在 #xmonad 的伙计们的鼓励下，我终于写了一个扩展来根据屏幕大小自动切换布局！这就是它：

```
{-# LANGUAGE FlexibleInstances, MultiParamTypeClasses #-}

-----------------------------------------------------------------------------
-- |
-- Module      :  XMonad.Layout.PerScreen
-- Copyright   :  (c) Edward Z. Yang
-- License     :  BSD-style (see LICENSE)
--
-- Maintainer  :  <ezyang@cs.stanford.edu>
-- Stability   :  unstable
-- Portability :  unportable
--
-- Configure layouts based on the width of your screen; use your
-- favorite multi-column layout for wide screens and a full-screen
-- layout for small ones.
-----------------------------------------------------------------------------

module XMonad.Layout.PerScreen
    ( -- * Usage
      -- $usage
      PerScreen,
      ifWider
    ) where

import XMonad
import qualified XMonad.StackSet as W

import Data.Maybe (fromMaybe)

-- $usage
-- You can use this module by importing it into your ~\/.xmonad\/xmonad.hs file:
--
-- > import XMonad.Layout.PerScreen
--
-- and modifying your layoutHook as follows (for example):
--
-- > layoutHook = ifWider 1280 (Tall 1 (3/100) (1/2) ||| Full) Full
--
-- Replace any of the layouts with any arbitrarily complicated layout.
-- ifWider can also be used inside other layout combinators.

ifWider :: (LayoutClass l1 a, LayoutClass l2 a)
               => Dimension   -- ^ target screen width
               -> (l1 a)      -- ^ layout to use when the screen is wide enough
               -> (l2 a)      -- ^ layout to use otherwise
               -> PerScreen l1 l2 a
ifWider w = PerScreen w False

data PerScreen l1 l2 a = PerScreen Dimension Bool (l1 a) (l2 a) deriving (Read, Show)

-- | Construct new PerScreen values with possibly modified layouts.
mkNewPerScreenT :: PerScreen l1 l2 a -> Maybe (l1 a) ->
                      PerScreen l1 l2 a
mkNewPerScreenT (PerScreen w _ lt lf) mlt' =
    (\lt' -> PerScreen w True lt' lf) $ fromMaybe lt mlt'

mkNewPerScreenF :: PerScreen l1 l2 a -> Maybe (l2 a) ->
                      PerScreen l1 l2 a
mkNewPerScreenF (PerScreen w _ lt lf) mlf' =
    (\lf' -> PerScreen w False lt lf') $ fromMaybe lf mlf'

instance (LayoutClass l1 a, LayoutClass l2 a, Show a) => LayoutClass (PerScreen l1 l2) a where
    runLayout (W.Workspace i p@(PerScreen w _ lt lf) ms) r
        | rect_width r > w    = do (wrs, mlt') <- runLayout (W.Workspace i lt ms) r
                                   return (wrs, Just $ mkNewPerScreenT p mlt')
        | otherwise           = do (wrs, mlt') <- runLayout (W.Workspace i lf ms) r
                                   return (wrs, Just $ mkNewPerScreenF p mlt')

    handleMessage (PerScreen w bool lt lf) m
        | bool      = handleMessage lt m >>= maybe (return Nothing) (\nt -> return . Just $ PerScreen w bool nt lf)
        | otherwise = handleMessage lf m >>= maybe (return Nothing) (\nf -> return . Just $ PerScreen w bool lt nf)

    description (PerScreen _ True  l1 _) = description l1
    description (PerScreen _ _     _ l2) = description l2

```

如果我能搞清楚他们该死的补丁提交流程，我会把它提交到 xmonad-contrib...
