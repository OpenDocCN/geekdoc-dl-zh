# 如何使用 Javascript 承诺延迟更新数据

> 原文：<https://winder.ai/how-to-use-javascript-promises-to-lazily-update-data/>

上周我在做一个简单的实现，为一个网站更新购物车，前端是用 html/javascript 写的。简介——当购物车中的商品数量被修改时，客户端可以按下 update cart 按钮来更新购物车数据库，之后需要重新计算订单的总值，并用新的总值刷新页面。

简单的解决方案是，遍历购物车中的商品，更新数据库并重新计算总数。然后刷新页面。

使用 Javascript 和 Array.forEach()方法

```py
function updateCart( items ) {
	items.forEach( function( item ) {
		updateToCart( item.id, item.quantity ); // update database  });
	// calculate the new totals  // refresh page } 
```

```py
function updateToCart( id, quantity ) {
    $.ajax( {
            url: "cart/update",
            type: "POST",
            data: JSON.stringify( {"id": id, "quantity": quantity} ),
            success: function ( data, textStatus, jqXHR) {
            console.log( 'Item updated: ' + id + ', ' + textStatus );
        },
        error: function ( jqXHR, textStatus, errorThrown ) {
            console.error( 'Could not update item: ' + id + ', due to: ' + textStatus + ' | ' + errorThrown);
        }
    });
} 
```

除了&mldr;

我们使用带有 jQuery 的 ajax 请求来更新购物车数据库。forEach 循环将遍历并完成，即使数据库仍在更新，或者更糟的是更新失败，代码的剩余部分也将被执行。

重新计算总数可能不正确，甚至更糟。这准确地演示了在 javascript 中何时不对数组使用 forEach 方法。

在 JAVASCRIPT 中小心使用数组。调用异步函数的 FOREACH()方法。

通常 javascript 中的 forEach 方法是同步的。但是，如果 forEach 方法的主体调用异步函数，那么 forEach 方法可以在异步函数完成之前完成。

为了避免这种不使用 forEach 方法的情况，创建一个回调函数来调用异步函数，并使用另一种方法来终止回调并继续正常的代码执行。一个 [if else 语句](https://developer.mozilla.org/Web/JavaScript/Reference/Statements/if...else)可以作为一种控制形式用于这种效果。

如果我们有一些需要同步执行的代码，并且其中有这种形式的 forEach 方法，那么删除 forEach 方法，并使用 if else 测试来拆分代码，以检查所有项何时被迭代。最好将代码分成两个函数。第二个函数只有在回调函数完成后才会被调用。

# 使用 Javascript 和带有 if &mldr; else 语句的数组

```py
// function updateCart() - replace items.forEach() with callback function next() function updateCart() {
    var cartsize = items.length;
    var idx = 0;
    next = function() {
        if (idx < cartsize) {
            // call updateToCart function passing the callback function next
 updateToCart(items[idx].id, items[idx].quantity, next);
            idx++; // increment the items counter
 } else {
            // continue execution
 recalcCartTotals();
        }
    }
    next(); // call the callback function to start the iteration } 
```

```py
function updateToCart( id, quantity, next ) {
    $.ajax( {
        url: "cart/update",
        type: "POST",
        data: JSON.stringify( { "id": id, "quantity": quantity } ),
        success: function ( data, textStatus, jqXHR ) {
            console.log( 'Item updated: ' + id + ', ' + textStatus );
            next();
        },
        error: function ( jqXHR, textStatus, errorThrown ) {
            console.error( 'Could not update item: ' + id + ', due to: ' + textStatus + ' | ' + errorThrown);
            next();
        }
    });
} 
```

# 更多最新发展…

使用 Javascript 回调函数一直是处理这些情况的公认方法，但是现在 Javascript 中有了一个新对象，Promise 对象，它是为处理异步操作而创建的。

[https://developer . Mozilla . org/en-US/docs/Web/JavaScript/Reference/Global _ Objects/Promise](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise)

在可迭代对象上使用的方法是 [Promise.all(可迭代对象)](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise/all)。

## 承诺承诺&mldr;

Promise.all()方法接受一个承诺数组，并在所有承诺都解决后触发一次回调:

*   promise 对象有一个 then()方法，允许您对 Promise 做出反应。

*   当承诺被解析时，将触发 then()方法回调。

*   第一个 then()方法回调接收 resolve()调用给它的结果:

为了实现 Promise 对象，它要求每次调用异步函数都返回 resolve 或 reject 回调。如果没有进行返回回调，那么 Promise 对象不会触发任何动作。

```py
function updateCart() {
	console.log("Updating Cart");
	var promises = [];
	var itemRows = document.getElementById("cart-list").rows;
	for(var i = 0; i < itemRows.length; i++) {
		var id = itemRows[i].cells[2].id;
		var quantity = ItemRows[i].cells[2].getElementsByTagName('input')[0].value;
		var p = new Promise(function(resolve, reject){updateToCart(id, quantity, resolve, reject);});
		promises.push(p);
	}
	Promise.all(promises).then(function() {
		recalcCartTotals();
	});
} 
```

```py
function updateToCart(id, quantity, resolve, reject) {
	console.log("Sending request to update cart: item: " + id + " 	quantity: " + quantity);
	$.ajax({
		url: "cart/update",
		type: "POST",
		data: JSON.stringify({"id": id, "quantity": quantity}),
		success: function (data, textStatus, jqXHR) {
			console.log('Item updated: ' + id + ', ' + textStatus);
			return resolve();
		},
		error: function (jqXHR, textStatus, errorThrown) {
			console.error('Could not update item: ' + id + ', due to: ' + textStatus + ' | ' + errorThrown);
			return reject();
		}
	});
} 
```

注意:不需要用 resolve 函数或 reject 函数返回任何值，但是如果需要的话，它会将错误的细节作为参数传递。