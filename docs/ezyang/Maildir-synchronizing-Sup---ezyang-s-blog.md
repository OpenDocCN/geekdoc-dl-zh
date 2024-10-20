<!--yml

分类：未分类

日期：2024-07-01 18:17:24

-->

# Maildir 同步 Sup：ezyang 的博客

> 来源：[`blog.ezyang.com/2012/12/maildir-synchronizing-sup/`](http://blog.ezyang.com/2012/12/maildir-synchronizing-sup/)

## Maildir 同步 Sup

在 Steven Hum 的推动下，我对我的 Sup 补丁集进行了一些最后的修饰，并将其“发布”到世界上（稍后会详细说明“发布”的含义）。该补丁集的总体主题是尽可能将 Sup 元数据与 Maildir 数据集成。具体来说：

+   它将 Damien Leone 的同步后补丁集与最新的 Sup 主线合并了起来。同步后补丁集可以将诸如“已读”或“已删除”等标志同步到 Maildir，然后可以使用 OfflineIMAP 将其传播回您的 IMAP 服务器。

+   此外，该补丁集具有同步任意标签的能力，具有一组简单的规则，根据消息的标签确定应将其移动到哪个文件夹。例如，收件箱和已归档的消息可以保存在单独的文件夹中，以便非 Sup 客户端可以有用地访问您关心的邮件。（相信我，这真的很棒。）这与一个实现快速远程消息移动的额外 OfflineIMAP 补丁配对使用。

+   它在 Maildir 上实现了 inotify，因此不再需要进行完整的目录扫描来检索新消息。轮询的瓶颈现在严格限制在 OfflineIMAP 上。

+   它实现了将已发送和草稿消息保存到 Maildir，以便它们显示在第三方客户端中。

+   最后，它还包含了一些我个人发现有用的杂项 bug 修复和额外的钩子。

由于我已经积极使用了一段时间，所以这个补丁集对您来说至少有很高的概率能够正常工作。Sup 有时会崩溃；如果这种情况不是可复现的或者不会导致数据丢失，我可能不会过于深入调查。我的一些补丁有点粗糙（特别是那些标记为 `HACK` 的：我已尝试在提交消息和代码注释中记录所有不规范的部分）。那么，这个版本的 Sup 有多少支持呢？嗯：

1.  因此，我在所有我关心的用例和环境中都在使用这个补丁集，它将继续正常工作；

1.  我可能不会修复我没有受影响的问题，绝对不会修复我无法复现的问题；

1.  我不保证一个稳定的提交历史：我已经多次重新基于这个补丁集，并将继续这样做。

但有些早期的补丁是相当不具争议性的，我希望它们最终能够进入主线。您可以在这里获取代码：[`gitorious.org/~ezyang/sup/ezyang/commits/maildir-sync/`](http://gitorious.org/~ezyang/sup/ezyang/commits/maildir-sync/)

### 新钩子

```
sent-save-to
  Configures where to save sent mail to. If this hook doesn't exist,
  the global sent setting will be used (possibly defaulting to sup://sent)
  Variables:
      message: RMail::Message instance of the mail to send.
      account: Account instance matching the From address
  Return value:
       Source to save mail to, nil to use default

compose-from
  Selects a default address for the From: header of a new message
  being composed.
  Variables:
    opts: a dictionary of ComposeMode options, including :from, :to,
      :cc, :bcc, :subject, :refs and :replytos
  Return value:
    A Person to be used as the default for the From: header

draft-save-to
  Selects a source to save a draft to.
  Variables:
    from_email: the email part of the From: line, or nil if empty
  Return value:
    A source to save the draft to.

```

### 标签同步

要使用此功能，在 `config.yaml` 中，您需要一个新选项 `:maildir_labels`：

```
:maildir_labels:
  :stanford: [[:inbox, 4], [null, 6]]

```

此选项的值是一个“账户”到“优先级列表”的字典。（账户标签 `stanford` 实际上并没有任何意义；它只是用于文档。）读取方法如下：

> 对于属于来源 4 或来源 6 的消息（请参阅`sources.yaml`），如果消息具有`:inbox`标签，则将其移动到来源 4；否则将其移动到来源 6。

这将自动开始对您更改标签的任何新邮件进行处理。为了将其应用于旧邮件，您需要运行`sup-sync-back-maildir`。如果您打算移动大量邮件，可能需要运行此版本的 OfflineIMAP：[`github.com/ezyang/offlineimap`](https://github.com/ezyang/offlineimap)
