# 使用 GoKit 的内聚微服务

> 原文：<https://winder.ai/cohesive-microservices-with-gokit/>

本周在开发交叉编排参考微服务应用时，我和来自 [Container-Solutions](https://container-solutions.com/) 和 [WeaveWorks](http://weave.works/) 的同事将我们所有的简单 Go 微服务移植到了 [GoKit 框架](https://gokit.io)。

对于简单的微服务，Go 是一个很好的语言选择。它的开销很低，并且有一个垃圾收集器。它比 C++速度更快，抽象程度更高。但是有[投诉](https://github.com/ksimka/go-is-not-good)。有趣的抱怨从“无法用谷歌搜索的名字”和“怪异的吉祥物”到“没有 OOP”和“不可否认”等相当合理的抱怨。

除了对语言的抱怨之外，可能还有三个因素减缓了“企业”的采用(这是否真的是一个问题，留待以后再说)。1)运营成本-开发和维护一个应用程序的成本和持续成本是多少(例如，找到开发人员有多容易？).2)工具——是否有足够的工具来支持企业开发(例如 CI/CD、IDE 等)。以及 3)框架支持——价值/时间比(例如，我们多快能开发出有价值的应用程序)。

GoKit 是一个 microserivces 框架，有一组习惯用法和一组库，旨在改进基于 Go 的应用程序的设计和灵活性。它并没有把*设定为*一个平台；相反，它试图将*整合到你选择的*平台中。我喜欢这种方式；它接受它所在的范围。应用程序级别，而不是流程编排。

## 从 Go 到 GoKit 的转换

(GoKit 示例)[https://gokit . io/examples/string SVC . html]是对 go kit 应用程序架构的极好介绍。它推广了洋葱模型，将业务关注点从中间件和传输层中分离出来。按照示例的工作流，生成一个具有大约四个类(大致相当于体系结构模型中的不同层)和一个大型主类的应用程序。大约有 350 行代码。等价的“哑”go 应用程序中的代码行数大约是 50。

从一开始，我第一个也是唯一一个关注的是代码行数的增加，而这几乎没有直接的价值。然而，我的担心很快就被减少的耦合和增加的内聚消除了。测试业务逻辑变得异常容易。

作为一个具体的例子，下面是创建和执行简单授权请求的代码:

```py
// Auth service interface
type Service interface {
	Authorise(total float32) (Authorisation, error)
}
// Auth service response struct
type Authorisation struct {
	Authorised bool `json:"authorised"`
}
// Auth service contstructor
func NewAuthorisationService(declineOverAmount float32) Service {
	return &service{
		declineOverAmount: declineOverAmount,
	}
}
// Auth service instance
type service struct {
	declineOverAmount float32
}
// Auth service implementation
func (s *service) Authorise(amount float32) (Authorisation, error) {
	if amount == 0 {
		return Authorisation{}, ErrZeroPayment
	}
	if amount < 0 {
		return Authorisation{}, ErrNegativePayment
	}
	authorised := false
	if amount <= s.declineOverAmount {
		authorised = true
	}
	return Authorisation{
		Authorised: authorised,
	}, nil
} 
```

有点啰嗦。比我希望的要多，但可能不会比 Java 差。但是请注意，传输实现(例如 HTTP)或工具(例如日志/度量)没有膨胀。这些被安置在别处。我们现在可以创建一个简单的测试来确保任何零支付请求都会产生一个错误:

```py
func TestFailIfAmountIsZero(t *testing.T) {
	_, err := NewAuthorisationService(10).Authorise(0)
	_, ok := err.(error)
	if !ok {
		t.Errorf("Authorise returned unexpected result: got %v want %v",
			err, "Zero payment")
	}
} 
```

在这里，GoKit 真的大放异彩。因为没有传输问题，所以不需要创建或模拟任何实现。当然，这种解耦也可以用其他框架来实现(比如 Spring)，但是在这里，它是由框架的架构来有效实施的。

## 免费的价值

GoKit 还附带了一系列运输支持和中间件。它支持 HTTP、Thrift 和 gRPC 开箱即用。但是真正的价值是由大量的中间件支持提供的。其中包括速率限制、日志记录、电路中断、负载平衡以及很多我还没有机会尝试的东西。再加上各种服务发现方案和监控系统支持，我很高兴地得出结论，这很容易产生我有幸尝试过的最好的 Go 微服务框架。

最后一点，我很幸运能和 GoKit 的作者 Peter Bourgon 一起工作。GoKit 未来版本中代码生成能力的承诺减轻了我对行数过多的担忧。他说，他希望“通过实现重复代码的自动生成，减少样板文件的数量(例如在服务定义中)”。我非常期待这次的加入。