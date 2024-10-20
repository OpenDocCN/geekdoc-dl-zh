<!--yml

类别：未分类

日期：2024-07-01 18:17:46

-->

# 比特币的密码学：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/06/the-cryptography-of-bitcoin/`](http://blog.ezyang.com/2011/06/the-cryptography-of-bitcoin/)

想要了解比特币使用的密码学，对于普通人来说实际上是非常困难的，如果不直接查看比特币的来源的话。例如，[opcode OP_CHECKSIG](https://en.bitcoin.it/wiki/OP_CHECKSIG) 明显是用来检查某些东西的签名……但没有明确说明它检查的是什么样的签名！（比特币中的操作码是什么？原来这个协议内建了一个非常巧妙的脚本系统用于构建交易。[你可以在这里了解更多。](https://en.bitcoin.it/wiki/Script)）所以实际上，我在我的文章[比特币不是去中心化的](http://blog.ezyang.com/2011/06/bitcoin-is-not-decentralized/)中弄错了一些事实上的细节，这是我在评论者 cruzer 声称，在密码哈希中断将只会降低挖矿难度，而不允许伪造交易时才意识到的。

所以我进行了研究并打开了比特币客户端的源代码。简短地说，我的论点的主要内容仍然相同，但是针对密码功能的假设攻击的细节则更加复杂——简单的选择前缀碰撞攻击是不够的。长篇大论？比特币在选择密码学方面做出了一些有趣的选择，本文的其余部分将探讨这些选择。比特币使用了两种哈希函数，[SHA-256](http://en.wikipedia.org/wiki/SHA-2) 和 [RIPEMD-160](http://en.wikipedia.org/wiki/RIPEMD)，但它还使用了椭圆曲线 DSA（[Elliptic Curve DSA](http://en.wikipedia.org/wiki/Elliptic_Curve_DSA)）在 secp256k1 曲线上执行签名。C++ 实现使用了 Crypto++ 库进行挖矿，使用 OpenSSL 进行普通用途。阅读本文的结尾，您将更好地理解比特币如何利用密码学模拟货币的属性。

### 比特币中的签名

在许多方面，这是比特币中的传统密码学。我们提出问题：“我们怎么知道 Alice 被授权将 100 比特币转给 Bob”，任何使用公钥密码学的人都知道答案是：“Alice 使用她的私钥对交易进行签名，并将此签名发布给比特币网络验证，使用她的公钥。” 这个签名是在 secp256k1 椭圆曲线上进行的（`key.h`）：

```
CKey()
{
    pkey = EC_KEY_new_by_curve_name(NID_secp256k1);
    if (pkey == NULL)
        throw key_error("CKey::CKey() : EC_KEY_new_by_curve_name failed");
    fSet = false;
}

```

[比特币社区已经讨论了椭圆曲线的选择](http://forum.bitcoin.org/?topic=2699.0)，似乎这个特定的曲线是为了可能的未来[速度优化](http://forum.bitcoin.org/index.php?topic=3238.0)而选择的。

然而，就像所有公共密码系统一样，比特币并不会对整个交易消息进行签名（那样将会非常昂贵）；相反，它对消息的密码哈希进行签名（`script.cpp`）：

```
uint256 SignatureHash(CScript scriptCode, const CTransaction& txTo,
                      unsigned int nIn, int nHashType)
{
    // ...
    // Serialize and hash
    CDataStream ss(SER_GETHASH);
    ss.reserve(10000);
    ss << txTmp << nHashType;
    return Hash(ss.begin(), ss.end());
}

```

这个哈希是 SHA-256 的*双重*应用：

```
template<typename T1>
inline uint256 Hash(const T1 pbegin, const T1 pend)
{
    static unsigned char pblank[1];
    uint256 hash1;
    SHA256((pbegin == pend ? pblank : (unsigned char*)&pbegin[0]), (pend - pbegin) * sizeof(pbegin[0]), (unsigned char*)&hash1);
    uint256 hash2;
    SHA256((unsigned char*)&hash1, sizeof(hash1), (unsigned char*)&hash2);
    return hash2;
}

```

好的，那我们如何破解呢？有几种方法：

+   我们可以破解底层的椭圆曲线加密，通过解决离散对数问题（这是量子计算机可以做的事情）或者破解选择的特定椭圆曲线来完成。在这个领域的大部分研究都是为了找出特定椭圆曲线中的漏洞，因此后者更有可能。

+   我们可以破解底层的加密哈希函数。在这种情况下，我们拥有一个我们想攻击的用户的已知签名，并生成另一个输入交易，使其哈希值相同，这样我们就可以重放先前的签名。这样的攻击将取决于比特币处理的序列化交易的形式：它对交易进行了一定量的处理，因此攻击者需要一些工作；然而，由于交易包括允许构建复杂交易的脚本系统，攻击者在构建这样一个输入时会有一定的余地。这种攻击无法对单次使用地址起作用，因为没有为重放而存在的这种签名。

破解签名算法需要选择性伪造攻击或更强大的攻击，这意味着任意交易可能被伪造并输入系统中。这将是一个完整的系统破解。对于签名重放攻击，可以通过添加客户端检查来确保相同的签名从未用于两个不同的交易，以增加一些保护措施。

### 比特币中的哈希算法

这是比特币中技术上新颖的加密使用方式，用来回答一个问题：“只有传统签名，爱丽丝可以无限次重新发送她实际上并不存在的比特币，有效地创建交易树的多个分支。我们如何防止这种情况？” 比特币提供的答案是：“交易链由解决计算难题（挖矿）的结果进行认证，一旦一个交易被包含在一个区块中确认，客户端更倾向于具有最高计算成本的交易链，使其他分支上的任何其他支出无效。” 即使你不相信去中心化货币，你也必须承认，这是相当优雅的解决方案。

更详细地说，计算难题本质上是对哈希函数的第一前像攻击的简化版本。矿工们得到一组解决方案哈希（所有零的哈希到目标哈希），并且需要找到一个具有特定结构的消息（一个区块链加上一个随机数），使其哈希为这些哈希中的一个。

在这种情况下，很容易看到哈希函数的首影像攻击（或者可能是稍弱攻击）意味着这个哈希问题可以更快地解决。如果对手知道这种方法但网络中没有人知道，这是一个安全漏洞；他可以轻易地占据超过 50%的网络计算能力并分裂区块链（记住：这是*指数*杠杆。我不在乎比特币网络有多少 Teraflops 的计算能力——聪明的算法总是赢）。在更严重的破坏中，他可以重建整个区块链来重写历史，执行足够的“计算工作”以说服网络上的其他客户端他的历史是真实的。这种攻击场景是众所周知的，并且在[这里描述](https://en.bitcoin.it/wiki/Weaknesses#Attacker_has_a_lot_of_computing_power)。请注意，一旦该方法被广泛传播并被其他矿工采用，计算能力的失衡将再次得到解决，并且哈希问题的难度可以相应地调整。

### 比特币地址

类似于 PGP 系统，比特币用户生成公钥和私钥对用于签名，但也发布一个便捷的“指纹”，实际上是一个 RIPEMD-160 哈希，供人们用作可以发送比特币到的标识符（`util.h`）：

```
inline uint160 Hash160(const std::vector<unsigned char>& vch)
{
    uint256 hash1;
    SHA256(&vch[0], vch.size(), (unsigned char*)&hash1);
    uint160 hash2;
    RIPEMD160((unsigned char*)&hash1, sizeof(hash1), (unsigned char*)&hash2);
    return hash2;
}

```

*与*像 PGP 这样的系统不同，比特币没有公钥分发机制：RIPEMD-160 哈希对于公钥是规范的。因此，如果在此密钥空间中发现碰撞，某人可能会从别人的地址中花费比特币。这种攻击场景在[此处描述](https://en.bitcoin.it/wiki/Address)。这种攻击通过比特币用户被鼓励为他们的钱包使用许多地址以及其他使用此碰撞能力的方式可能对攻击者更有利（如上所述）来减轻。

### 结论

如我们所见，多种不同的密码原语在集成中用于指定比特币协议。一个原语的妥协不一定会影响系统的其他部分。然而，所有这些原语都被硬编码到比特币协议中，因此我在[我以前的文章](http://blog.ezyang.com/2011/06/bitcoin-is-not-decentralized/)中提出的论点仍然成立。
