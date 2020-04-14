## Markdown
- title
    Use "# ## ###..." to create level title  
- list
    - Use "- " to create non-ordered list
    - Use "1. 2. 3." to create ordered list  
- new line
    - Use "2 more SPACE and ENTER" to get a new line
- quote
    - Use ">" to create a quote 
        > So I think, So I am.
- italic & bold
    - Use "\*abc\*" to create italic text 
        such as *italic*
    - Use "\*\*abc\*\*" to create bold text
        such as **bold**
- URL & picture
    - Use "\[]()" to set a URL
        such as [google](https://www.google.com)
    - Use "\!\[]()" to set a pictrue
      such as ![Markdown](http://mouapp.com/Mou_128.png)
- code
    Use "\`code`" to create code box  

    such as `print("hello world");`
- deviding line
    Use "***" create deviding line
     ***
- table
    Use 
    "\| ... \|" 
    "------"
    "\| ... \|" 
    to create table
    such as 

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|	
| col 3 is      | right-aligned |  $16  |

## git
**add -> commite -> push**
- init
`git init`
- add
`git add filename`
`git add .`
`git add --all`
- remove
`git rm filename`
- commit
`git commit -m "notes"`
- status
`git status`
`git diff filename`
`git log`    // View log history
`git reflog` // View commond history
- reset
    1. `git reset --hard HEAD^`
    2. `git reset --hard d7b5`
- branch
`git branch` // branch overview
`git merge branch1` // merge branch1 to now-branch
`git checkout branchname`// switch branch
    - -b create & switch
    - -d delete
- tag
`git tag`
`git tag tagname d7b5`
`git show tagname`
- push
`git push origin`
- remote repository
`git remote add origin git://127.0.0.1/abc.git`
`git remote remove origin`
`git push -u origin master`
`git pull` // Tips: if conflict, first pull, solve the confliction, then push again

- 遇到的坑: 我的电脑上的ssh public key绑定了dy-chenzq这个repo, 导致我想将我的Note push到github上时出现了"key is already in used错误"
解决方法: 
  1. 删除之前给项目绑定的ssh key, 用账户全局key代替.
  2. 新建ssh key, 建立config, 分别绑定.

## 计算机网络基础

### OSI七层模型 & TCP/IP五层模型 
- TCP/IP(物理层+数据链路层=网络接口层)
  物理层 -> 数据链路层 -> 网络层 -> 传输层 -> 应用层
- OSI
  物理层 -> 数据链路层 -> 网络层 -> 传输层 -> **会话层 -> 表示层 -> 应用层**
  1. 物理层
     确保原始数据在可靠的物理媒介上传输
     e.g. 以太网 令牌环  
     
  2. 数据链路层
     - 在不可靠的物理介质上为网络层提供可靠的数据传输服务
     - 基本数据单位为帧
     - 主要的协议: 以太网协议
     - 两个重要设备名称: 网桥 交换机  
     ***Note***: "网桥"即集线器, 根本目的是延伸网线, 网桥+MAC地址学习≈交换机(避免了对全体的帧广播)
     
  3. 网络层
    - 提供不可靠, 无连接的传送服务
    - 路径选择, 路由, 逻辑寻址
    - 主要协议
      - IP(Internet Protocol, 互联网互联协议)
      - ICMP(Internet Control Message Protocol, 因特网报文控制协议) *Ping*
      - ARP/RARP(\[Reverse] Address Resolution Protocol, 地址解析协议)
    - 重要的设备: 路由器

  4. 传输层
    - 提供端到端的可靠或不可靠的传输, 差别控制和流量控制
    - 主要协议
      - TCP(Transmission Control Protocol, 传输控制协议)
      - UDP(User Datagram Protocol, 用户数据报协议)
    - 重要设备: 网关
  5. 会话层
    负责建立, 管理, 终止进程间的**会话**
  6. 表示层
    对数据进行**转换**, 包括加密、压缩、格式转换等
  7. 应用层
    - 为操作系统提供访问网络服务的**接口**
    - 主要协议
      - FTP(文件传输协议)
      - Telnet(远程登录协议)
      - DNS(域名解析)
      - SMTP(邮件传输协议)
      - POP3(邮件协议)
      - HTTP(超文本传输协议)


### IP地址
- A类地址 
  0.0.0.0 ~ 127.255.255.255, 第一个字节作为网络号
- B类地址
  128.0.0.0 ~ 191.255.255.255, 前两个字节作为网络号
- C类地址
  192.0.0.0 ~ 223.255.255.255, 前三个字节作为网络号
- D类地址
  224.0.0.0 ~ 239.255.255.255
- 回环地址
  127.0.0.0/8  
  常用127.0.0.1
- 子网掩码
  如果两个IP地址在子网掩码的按位与的计算下所得结果相同，即表明它们共属于同一子网中  
  全为0时代表本地网络, 全为1时代表广播网络

### ARP/RARP
- ARP
  - IP地址->MAC地址  
  - 根据IP地址获取MAC地址的协议, 网络中的每台主机收到ARP广播后将会将其存入缓存中一段时间  
  - 流程
    1. A在本地路由表上查找是否有B的IP-MAC信息
    2. 若无, 则对全体广播
    3. 接收广播的主机检查广播的接收IP是否是自己的IP
    4. 如果是自己的, 则回应ARP请求
- RARP
MAC地址->IP地址

### 路由选择协议
- RIP协议
  底层为Bellman-Ford算法(对图做V-1次*松弛*操作, 权值可以为负数, 但时间复杂度高达O(VE)), 当跳数高于15时则会丢弃数据包
  > 松弛: 通过BFS确定A到其他各点的最短路径
- OSPF协议(Open Shortest Path First)
  底层为Dijkstra算法(贪心, 不断加入顶点并维护dis数组, 直到所有顶点都加入了图中, 每次加入后更新数组, 时间复杂度O(V*V)), 选择路由的度量标准是带宽, 延迟
  
### TCP/IP协议
可靠, 面向连接的协议, 由网络层的IP协议和传输层的TCP协议组成. TCP负责控制传输, IP负责给每一台设备规定一个地址. 
- 三次握手
- 四次挥手
![pic](https://www.runoob.com/wp-content/uploads/2018/09/1538030297-7824-20150904110008388-1768388886.gif)

- 使用TCP的协议
  - FTP
  - Telnet(远程登录协议)
  - SMTP(简单邮件传输协议)
  - HTTP

### UDP协议
不可靠, 面向无连接的协议, 可广播发送, 主要应用于查询--应答服务
- 使用UDP的协议
  - TFTP(简单文件传输协议)
  - SNMP(简单网络管理协议)
  - DNS(域名解析协议)

### DNS协议
(Domain Name System, 域名系统), 将URL转换为IP地址

### NAT协议
(Network Address Translation, 网络地址转换)将内网地址转化为外网地址, 解决了IP不足和外部攻击的问题

### DHCP协议
(Dynamic Host Configuration Protocol, 动态主机设置协议), 应用于局域网, 使用UDP工作, 给内网自动分配IP地址, 并管理内网

### HTTP协议
(HyperText Transfer Protocol, 超文本传输协议)  
其请求包括
- GET  
  从服务器上读取URL标示的信息
- POST
  给服务器添加信息
- PUT
  在给定的URL下存储一个文档
- DELETE
  删除指定的资源

**NOTE**:GET和POST的区别:
- GET是从服务器获取数据, POST这是上传数据
- GET可以将参数加到URL中
- GET传送的数据量小(<2kb), POST较大
- GET应该是安全的(不产生副作用), 幂等的(同一多个GET返回的结果相同)

*EXAMPLE*
输入cn.bing.com后执行的全过程
1. 应用层  
 **DNS**解析, URL->IP, 得到 202.89.233.100, 浏览器发起**HTTP**会话到此IP, 通过**TCP**封装数据包
2. 传输层  
把HTTP请求分成报文段, 添加端口号, 使用IP地址查找目的端
3. 网络层  
通过路由选择算法选择路径到达服务器
4. 链路层  
通过**ARP**广播找到服务器的MAC地址, 服务器应答后即可开始传输















