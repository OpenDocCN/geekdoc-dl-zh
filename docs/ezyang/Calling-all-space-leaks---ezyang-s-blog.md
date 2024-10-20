<!--yml

category: 未分类

date: 2024-07-01 18:17:47

-->

# -   Calling all space leaks：ezyang's 博客

> 来源：[`blog.ezyang.com/2011/05/calling-all-space-leaks/`](http://blog.ezyang.com/2011/05/calling-all-space-leaks/)

## Calling all space leaks

I’m currently collecting non-stack-overflow space leaks, in preparation for a future post in the Haskell Heap series. If you have any interesting space leaks, especially if they’re due to laziness, send them my way.

Here’s what I have so far (unverified: some of these may not leak or may be stack overflows. I’ll be curating them soon).

```
import Control.Concurrent.MVar

-- http://groups.google.com/group/fa.haskell/msg/e6d1d5862ecb319b
main1 = do file <- getContents
           putStrLn $ show (length $ lines file) ++ " " ++
                      show (length $ words file) ++ " " ++
                      show (length file)

-- http://www.haskell.org/haskellwiki/Memory_leak
main2 = let xs = [1..1000000::Integer]
        in print (sum xs * product xs)

-- http://hackage.haskell.org/trac/ghc/ticket/4334
leaky_lines                   :: String -> [String]
leaky_lines ""                =  []
leaky_lines s                 =  let (l, s') = break (== '\n') s
                                 in  l : case s' of
                                              []      -> []
                                              (_:s'') -> leaky_lines s''

-- http://stackoverflow.com/questions/5552433/how-to-reason-about-space-complexity-in-haskell
data MyTree = MyNode [MyTree] | MyLeaf [Int]

makeTree :: Int -> MyTree
makeTree 0 = MyLeaf [0..99]
makeTree n = MyNode [ makeTree (n - 1)
                  , makeTree (n - 1) ]

count2 :: MyTree -> MyTree -> Int
count2 r (MyNode xs) = 1 + sum (map (count2 r) xs)
count2 r (MyLeaf xs) = length xs

-- http://stackoverflow.com/questions/2777686/how-do-i-write-a-constant-space-length-function-in-haskell
leaky_length xs = length' xs 0
  where length' [] n = n
        length' (x:xs) n = length' xs (n + 1)

-- http://stackoverflow.com/questions/3190098/space-leak-in-list-program
leaky_sequence [] = [[]]
leaky_sequence (xs:xss) = [ y:ys | y <- xs, ys <- leaky_sequence xss ]

-- http://hackage.haskell.org/trac/ghc/ticket/917
initlast :: (()->[a]) -> ([a], a)
initlast xs = (init (xs ()), last (xs ()))

main8 = print $ case initlast (\()->[0..1000000000]) of
                 (init, last) -> (length init, last)

-- http://hackage.haskell.org/trac/ghc/ticket/3944
waitQSem :: MVar (Int,[MVar ()]) -> IO ()
waitQSem sem = do
   (avail,blocked) <- takeMVar sem
   if avail > 0 then
     putMVar sem (avail-1,[])
    else do
     b <- newEmptyMVar
     putMVar sem (0, blocked++[b])
     takeMVar b

-- http://hackage.haskell.org/trac/ghc/ticket/2607
data Tree a = Tree a [Tree a] deriving Show
data TreeEvent = Start String
                | Stop
                | Leaf String
                deriving Show
main10 = print . snd . build $ Start "top" : cycle [Leaf "sub"]
type UnconsumedEvent = TreeEvent        -- Alias for program documentation
build :: [TreeEvent] -> ([UnconsumedEvent], [Tree String])
build (Start str : es) =
        let (es', subnodes) = build es
            (spill, siblings) = build es'
        in (spill, (Tree str subnodes : siblings))
build (Leaf str : es) =
        let (spill, siblings) = build es
        in (spill, Tree str [] : siblings)
build (Stop : es) = (es, [])
build [] = ([], [])

```
