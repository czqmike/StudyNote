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

- math
[markdown_math](https://www.jianshu.com/p/e74eb43960a1)
[markdown_matrix](https://blog.csdn.net/qq_38228254/article/details/79469727)
  - superscript & subscript
  Use "^" / "_" to set a super / sub script, multi script needs been quoted by "{}"
  such as $X^4$ $X_1$ $Y_{12}$
  

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

- 遇到的坑: 我的电脑上的ssh public key绑定了另一个repo, 导致我想将我的Note push到github上时出现了"key is already in used错误"
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
  四次挥手的几个状态需要注意一下：
  1. 客户端
    `Fin-Wait-1`, `Fin-Wait-2`, `Time-Wait` 
    - `Fin-Wait-1`：
      客户端主动关闭，等待服务端接收到关闭`ACK`
    - `Fin-Wait-2`：
    客户端主动关闭，等待服务端接收到关闭`FIN`

    - `Time-Wait`包含两个`MSL` (Maximun Segment Lifetime, defalut = `30s`)
      等待两个`MSL`的好处：[ref](https://blog.51cto.com/wangziyin/1302668)
        1. 保证服务端能收到第四次挥手的ACK，结束连接
        2. 防止已失效的报文出现在此次连接中 
  2. 服务器端
    `Close-Wait`
      这种状态的含义其实是表示在**等待关闭**。怎么理解呢？当对方close一个SOCKET后发送FIN报文给自己，你系统毫无疑问地会回应一个ACK报文给对方，此时则进入到CLOSE_WAIT状态。接下来呢，实际上你真正需要考虑的事情是察看你是否还有数据发送给对方，如果没有的话，那么你也就可以close这个SOCKET，发送FIN报文给对方，也即关闭连接。所以你在CLOSE_WAIT状态下，需要完成的事情是等待你去关闭连接。

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


## 音视频编解码基础
包括封装技术、视频压缩编码、音频压缩编码和流媒体传输协议。  

**播放网络上的视频文件时的流程**：
![播放视频流程图](https://img-blog.csdn.net/20140201120523046?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGVpeGlhb2h1YTEwMjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
1. 解协议(**HTTP, RTMP**, MMS etc.)
   去除协议中的信号指令数据，**只保留音视频数据**
2. 解封装(**MP4, RMVB, AVI** etc.)
   将压缩编码的音视频数据**分离**成音频流和视频流
3. 解码音视频(**AAC, H264** etc.)
   将压缩的音视频数据解码变成**非压缩**的音视频数据.  
   解码后, 视频数据变为**RGB, YUV420P**等颜色数据, 音频数据变成音频采样数据
4. 音视频同步
   同步解码出来的音视频流

### 流媒体协议
|名称|传输层协议|客户端|使用领域|
|:--|:-----------|:-------|:-----|
|RTSP+RTP|TCP+UDP|VLC, WMP|IPTV|
|RTMP|TCP|Flash|互联网直播|
|RTMFP|UDP|Flash|互联网直播+点播|
|HTTP|TCP|Flash|互联网点播|

RTSP+RTP常用于IPTV, 因为UDP可采用组播, 效率高, 缺点是网络不好的时候可能产生丢包, 影响观看质量  

HTTP, RTMP这类服务因采用TCP作为传输协议, 不会发生丢包, 保证了视频的质量, 被广泛应用于互联网视频服务中

### 封装格式
|名称|流媒体支持|支持的视频格式|支持的音频编码|使用领域|
|:----|:-----|:-----------|:-----------|:------|
|AVI|NO |ALL               |AAC, MPEG-1 etc.|BT下载|
|MP4|YES|MPEG-4, H.264 etc.|AAC, MPEG-1 etc.|互联网视频|
|TS |YES|MEPG-4, H.264 etc.|AAC, MPEG-1 etc.|IPTV|
|FLV|YES|VP6, H.264 etc.   |AAC, MP3 etc.   |互联网视频|
|MKV|YES|ALL               |ALL             |互联网视频|

### 视频编码
[视频编码的基本原理](https://blog.csdn.net/leixiaohua1020/article/details/28114081)
由于数据冗余和视觉冗余的存在, 使视频数据可以得到极大的压缩.一般将变换编码, 运动估计和运动补偿以及熵编码三种方式结合使用, 共同进行压缩编码

|名称|推出时间|使用领域|
|:--|:------|:-----|
|HEVC(H.265)|2013|研发中|
|H.264|2003|各个领域|
|MPEG4|2001|不温不火|
|VP9|2013|研发中|

编码格式比较:HEVC > VP9 > H.264> VP8 > MPEG4 > H.263 > MPEG2。

#### H264简析
H.264原始码流（又称为“裸流”）是由一个一个的NALU组成的。他们的结构如图所示:
![NALU结构图](https://img-blog.csdn.net/20160118001549018)
其中每个NALU之间通过startcode（起始码）进行分隔，起始码分成两种：0x000001（3Byte）或者0x00000001（4Byte）。如果NALU对应的Slice为一帧的开始就用0x00000001，否则就用0x000001。
H.264码流解析的步骤就是首先从码流中搜索0x000001和0x00000001，分离出NALU；然后再分析NALU的各个字段。

#### YUV & RGB
一般的视频采集芯片输出的码流都是YUV数据流, 而H.264, MPEG的编解码也是在原始的YUV码流上进行编码和解析.  
YUV分为Y(Luminance, 明亮度), 描述灰度值, UV(Chrominance, 色度), 指定像素的颜色.如果只有Y, 无UV, 则图像也能显示, 但是只能显示为黑白图像. YUV不想RGB那样要求三个独立的视频信号同时传输, 所以占用极少的频宽.
因为人眼感受亮度的细胞比感受颜色的细胞更加多，所以相比于颜色，人对于亮度更加敏感。
基于这个原理，我们可以对图像进行压缩，就产生了YUV420
Y, U, V, R, G, B $\in[0, 255]$
- YUV420
  对每行扫描线来说，只有一种色度分量`U: V`以2:1的抽样率存储。相邻的扫描行存储不同的色度分量，也就是说，如果一行是4:2:0的话，下一行就是4:0:2，再下一行是4:2:0...以此类推。
  *EXAMPLE*
  原始像素:
  [Y0 U0 V0] [Y1 U1 V1] [Y2 U2 V2] [Y3 U3 V3]
  [Y5 U5 V5] [Y6 U6 V6] [Y7U7 V7] [Y8 U8 V8]
  存放的码流为：
  Y0 U0 Y1 Y2 U2 Y3
  Y5 V5 Y6 Y7 V7 Y8
  映射出的像素点为：
  [Y0 U0 V5] [Y1 U0 V5] [Y2 U2 V7] [Y3 U2 V7]
  [Y5 U0 V5] [Y6 U0 V5] [Y7U2 V7] [Y8 U2 V7]
  压缩比为：`2:1`
         
RGB则采用红绿蓝三种颜色以不同强度混合来表示图像. 红、绿、蓝三盏灯的叠加情况，中心三色最亮的叠加区为白色，加法混合的特点：越叠加越明亮。

**YUV 与 RGB的互相转换**
$
\begin{bmatrix}
Y \\
U \\
V \\
\end{bmatrix} = 
\begin{bmatrix}
0.299 & 0.587 & 0.114 \\
-0.169 & -0.331 & 0.5 \\
0.5 & -0.419 & -0.081
\end{bmatrix}
\begin{bmatrix}
R \\
G \\
B \\
\end{bmatrix} +
\begin{bmatrix}
0 \\
128 \\
128 \\
\end{bmatrix}
$

$
\begin{bmatrix}
R \\
G \\
B \\
\end{bmatrix} = 
\begin{bmatrix}
1 & -0.00093 & 1.401687 \\
1 & -0.3437 & -0.71417 \\
1 & 1.77216 & 0.00099 
\end{bmatrix}
\begin{bmatrix}
Y \\
U-128 \\
V-128 \\
\end{bmatrix}
$

### 音频编码
|名称|推出时间|使用领域|
|:--|:------|:-----|
|AAC|1997|各个领域|
|MP3|1993|早期|
|WMA|1999|微软|

近些年来音频编码格式无较大创新, 说明现有的音频编码技术已经大体上满足了人们的需求.

## GStreamer
### CommondLine
命令行无法运行， 或缺少插件？
### gst in C
- Basic tutorial 1
  1. init
    gst_init(NULL, NULL);
  2. launch
    pipeline = gst_parse_launch(URI, NULL);
  3. set state
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    bus = gst_element_get_bus(pipeline);
  4. loop
    msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
          GST_MESSAGE_ERROR | GST_MESSAGE_EOS);
  5. clean up
    gst_message_unref()
    gst_object_unref()
- Basic tuturial 2
  1. create elements
    gst_element_factory_make()
  2. create empty pipeline
    gst_pipeline_new()
  3. add elements to pipeline
    gst_bin_add_many()
  4. link elements to each other
    gst_element_link()
### GObject
- 封装
  在GObject中, 类是实例结构体与类结构体的组合, 类结构体一般只被初始化一次, 而实例结构体初始化次数等于对象实例化次数.  
  类数据(静态数据保存在类结构体中), 而对象数据则保存在实例结构体中.  
  ![example_code](https://img-my.csdn.net/uploads/201207/24/1343097249_6961.png)
  使用G_DEFINE_TYPE(GUPnPContext, gupnp_context, 
                   GSSDP_TYPE_CLIENT);
                   进行类的定义.
- 继承
  GObject通过在gupnpcontext实例中声明GSSDPClient parent来告知GObject系统GSSDPClinet是gupnpcontext的双亲, 同时通过在定义中声明  
  GSSDPClientClassparent_class进行类结构体和实例结构体的共同声明.
- 多态
  GObject通过在每个子类的内存中保存了成员函数指针的虚方法表来实现多态, 在运行时会查找合适的函数指针进行覆盖.
### DEBUG
  - 设置环境变量GST_DEBUG
  - Debug function
    GST_ERROR(), GST_WARNING(), GST_INFO(), GST_LOG(), GST_DEBUG()
    e.g.
    > To change the category to something more meaningful, add these two lines at the top of your code:
      `GST_DEBUG_CATEGORY_STATIC (my_category);`
      `#define GST_CAT_DEFAULT my_category`
    > And then this one after you have initialized GStreamer with gst_init():
      `GST_DEBUG_CATEGORY_INIT (my_category, "my category", 0, "This is my very own");`
  - 设置GST_DEBUG_DUMP_DOT_DIR并使用GST_DEBUG_BIN_TO_DOT_FILE(), 
    GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS()去产生pipeline graph
### Plugins
- shm: shmsink & shmsrc
  [shm doc](https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gst-plugins-bad/html/gst-plugins-bad-plugins-plugin-shm.html)

  shm can share memory between progresses, it use a `shmsink` to push data and 
  a `shmsrc` to get data. The memory is been set as property `socket-path`.

  Here is a test command line below:
  ```bash
  ## Start shmsink
  gst-launch-1.0 videotestsrc ! x264enc ! shmsink socket-path=/tmp/foo sync=true wait-for-connection=false shm-size=10000000

  ## Start shmsrc
  gst-launch-1.0 shmsrc socket-path=/tmp/foo ! h264parse ! avdec_h264 ! videoconvert ! ximagesink
  ```


## FFmpeg
[操作文档](http://www.ruanyifeng.com/blog/2020/01/ffmpeg.html)
- 查看视频元信息  
  `ffmpeg -i input.mp4 [-hide_banner]`
- 转换编码格式
  `ffmpeg -i input.mp4 -c:v libx264 output.mp4` 
- 转换容器格式(不改变编码)
  `ffmpeg -i input.mp4 -c copy output.webm`
- 调整码率
  `ffmpeg -i input.mp4 \  

  -minrate 964K -maxrate 3856K -bufsize 2000K \  

  output.mp4`
- 改变分辨率
  `ffmpeg -i input.mp4 -vf scale=480:-1 output.mp4`
- 裁剪(提取新片断)
  `ffmpeg -ss [start] -i [input] -t [duration] -c copy[output]` 
  `ffmpeg -ss [start] -i [input] -to [end] -c copy [output]`
  e.g.
  `ffmpeg -ss 00:01:50 -i [input] -t 10.5 -c copy [output]`
  `ffmpeg -ss 2.5 -i [input] to 10 -c copy [output]`

  ## BBR
  ### BBR概述
  BBR(Bottleneck Bandwidth and RTT)是基于接收端反馈和发送端调节码率的拥塞控制算法。
  > Bandwidth(带宽), RTT(Round Trip Time, 往返时延)

  - BBR-BDP
    BDP(Bandwidth Delay Product, 拥塞控制窗口)代表链路上可以存放的最大数据容量.
    BDP = min_RTT * max_bandwidth  

  *** 

  ### 四种状态机
  BBR通过4种状态机来探测RTT和Bandwidth.
  - Startup
    慢启动, 以固定增益系数( $2/\ln2$ )增大发包速率, 若有一轮最大瓶颈带宽没有增加25%以上, 则说明带宽已满.
  - Drain
    把Startup中的数据包排空.
  - ProbeBW
    稳定状态, 说明已经测得最大带宽, 每个包都不会产生排队现象, 大部分时间都在此状态.
  - ProbeRTT
    前三种状态都可能进入ProbeRTT状态, 如果超过10s未检测到更小的RTT, 则进入此状态: 把发包量降低, 测得一个较为准确的RTT.
    测完后若带宽是满的, 则进入Startup状态, 若不满则进入ProbeBW状态.
  ![BBR的4种状态机](https://mmbiz.qpic.cn/mmbiz_png/ibMNOojSZERvWaG3XYEAZPajoQEEQ9PicnMUptJaicQnJVk7AkjoR75m0vicyn3LJHZvVCaFoLHWMu8Ij9HiaCfBElw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### BBR概述
[BBR算法详解](https://blog.csdn.net/dog250/article/details/52830576)
![总体流程](https://img-blog.csdn.net/20161016152908882?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

1. 即时带宽的计算
   BBR作为一个纯粹的拥塞控制算法，完全忽略了系统层面的TCP状态，计算带宽时它仅仅需要两个值就够了：
  1).应答了多少数据，记为**delivered**；
  2).应答1)中的delivered这么多数据所用的时间，记为**interval_us**。
  将上述二者相除，就能得到带宽：
    > bw = delivered / interval_us

2. 状态机的作用
   4个状态机并不影响bw的计算, 仅仅影响**增益系数**的计算. 增益系数 > 1, 则可以比当前计算的bw发送更多的数据, 反之, pacing rate和cwnd要比bw计算的结果要小. 
   在稳定状态中, 我们照常计算当前的bw, 但在计算pacing rate和cwnd的时候, 使用[5/4, 3/4, 1, 1, 1, 1]这样一个常系数列来维持bw和RTT缓慢变化.

3. pacing rate 与 cwnd
   pacing rate 与 cwnd是BBR的输出参数, cwnd规定了当前的TCP可以发送多少数据, pacing rate规定了发送这些数据的时间间隔, 以免堵塞网络. 
   - 计算:  
   设10轮采样内最大的bw为BW, 设当前的增益系数为G, 当前的cwnd增益系数为G' 则
     > pacing rate = BW * G
     > BDP = min_RTT * max_BW
     > cwnd = BDP * G'
     
### BBR-BitrateAdjuster实现
> 当前的实现: 测量BW和RTT, 然后根据BDP队列的平均值和AIMD来调整bitrate
- 问题 
  AIMD降得太快了, 大部分时间都在不停地调整, 45度角形成的三角形会造成最多50%的带宽浪费.
- 分析
  BBR本身运行在系统内核级别, 采用C实现, 可以达到统计每个ACK的精准调控.
  而我们这个模块只能跑在推流应用层, 并且每N秒才有机会重新检测调整, 所以精度上会差一些.
  现在BW和RTT本身的测量没有太大问题, 问题在于应该**改变Adjuster的调整策略**.
  因为在BBR中4种状态机的存在是为了调整增益系数, 并不参与对BDP的计算中, 所以如果移植BBR, 无需对NetworkPeeker类做过多调整, 只需要在Adjuster中实现4种状态机, 再由这四种状态机完成*增益系数G*的计算, 由G去调整bitrate即可.
  bitrate可以参照BBR的输出: pacing rate & cwnd

- 四种状态机的详细参数
  1. STARTUP
     当BW增长缓慢时停止, 
     > pacing_gain = 2.89, cwnd_gain = 2.89
  2. DRAIN
     释放在STARTUP中拥堵的包, 
     > pacing_gain = 1.35 = 1/2.89, cwnd_gain = 2.89
  3. PROBE_BW
     循环常数列pacing_gain来维持合适的BW (cwnd_gain === 2)
     > pacing_gain = [1.25, 0.75, 1, 1, 1, 1, 1, 1]
  4. PROBW_RTT
     如果需要, 以更慢的发送速率去适应RTT
     > pacing_gain = 1.0


     

输入: 当前数据包状态
输出: bitrate
1. 测量模块
   构造4种状态机, 测量出min_RTT和max_BW (对数据包的处理拟使用scapy)
   问题:
   - scapy的抓包与记录包状态 (√)
   - 如何判断接收到的dump的时间来计算BW?
   - 如何控制排空wipe来计算RTT?
   - 如何计算增益系数G?
2. 调整模块
   根据min_RTT和max_BW, 使用AIMD动态地调整bitrate
   `h264enc.set_property("bitrate", bitrate)`

### TC - Network Emulator
tc: netem 是 一个网络模拟功能模块。该功能模块可以用来在性能良好的局域网中,模拟出复杂的互联网传输性能,诸如低带宽、传输延迟、丢包等等情况。
TC通过在输出端口处建立一个队列来实现流量控制.

TC只能控制**发包**动作,不能控制收包动作,同时,它直接对**物理接口**生效,如果控制了物理的 eth0,那么逻辑网卡(比如 eth0:1)也会受到影响,反之,如果在逻辑网卡上做控制,该控制可能是无效的。

[How to Use the Linux Traffic Control](https://netbeez.net/blog/how-to-use-the-linux-traffic-control/)

*Example*
1. 查看你的网卡信息
  `ifconfig`
2. 将网卡加入监控列表 
  `sudo tc qdisc add dev eth0 root netem`
3. 设置参数
  - 设置延时
  `sudo tc qdisc add dev eth0 root netem delay 200ms`
  - 设置带宽
  `sudo tc qdisc add dev enp5s0f1 root netem rate 300kbit 20 100 5`
  - 设置丢包率 
  `sudo tc qdisc change dev eth0 root netem loss 0.5% `
  - 设置重发
  `sudo tc qdisc change dev eth0 root netem duplicate 1%`
  - 设置发乱序包
  `sudo tc qdisc change dev eth0 root netem gap 5 delay 10ms`
4. 展示已应用的规则
  `sudo tc qdisc show dev eth0`
5. 删除全部规则
  `sudo tc qdisc del dev eth0 root`

**Note**:
if you have an existing rule you can change it by using `tc qdisc change…` and if you don’t have any rules you add rules with ` tc qdisc add...`

## ZLMedia踩坑
- 如何安装部署？
  `git clone --depth 1 https://gitee.com/xiahcu/ZLMediaKit`
  `cd ZLMediaKit`
  `git submodule update --init`
  `sh sh build_for_linux.sh`
- 如何启动运行？
  > 启动Server
  `sudo ~/ZLMediaKit/release/linux/Debug/MediaServer`
  `sudo ~/ZLMediaKit/release/linux/Debug/MediaServer --ssl ~/ZLMediaKit/release/linux/Debug/czqmike-server.cn.pem` 
  > 推流RTSP
  `sudo ffmpeg -re -stream_loop -1 -i "/home/czq/Videos/1280.mp4" -vcodec h264 -acodec aac -f rtsp -rtsp_transport tcp rtsp://192.168.216.3/live/test`
  > 推流RTMP
  `sudo ffmpeg -re -stream_loop -1 -i "/home/czq/Videos/1280.mp4" -vcodec h264 -acodec aac -f flv rtmp://192.168.216.3/live/test`
  > 推流RTMPS
  `sudo ffmpeg -re -stream_loop -1 -i "/home/czq/Videos/1280.mp4" -vcodec h264 -acodec aac -f flv rtmps://czqmike-server.cn:19350/live/test`
- 推流RTMP 1280P卡顿如何解决
  虚拟机CPU资源过少，添加CPU资源可解决

- 如何应用SSL加密RTMP
  1. [创建SSL证书](https://www.jianshu.com/p/eaad77ed1c1b)
  2. 在启动选项后加上 -s path-to-ssl.pem
  3. [在Server上安装证书](https://www.cnblogs.com/rader/p/3781880.html)
  
  **FAIL**
  [tls @ 0x55844f4ea160] A TLS fatal alert has been received.
  [rtmps @ 0x55844f4e12a0] Cannot open connection tls://192.168.216.3:19350
  rtmps://192.168.216.3:19350/live/test: Input/output error


- 性能对比SRS
  [详细测试报告](https://github.com/xiongziliang/ZLMediaKit/wiki/Benchmark)
  ZLMedia 推RTMP时, CPU占用总体上为SRS的2倍左右, 内存占用则为SRS的2.3倍左右 

## SRS学习 
### 部署
`cd srs && docker build -t srs .`

### 启动本地SRS服务
`docker run --rm -it --net=host srs bash`
`cd /usr/local/srs && ./objs/srs -c conf/srs.conf`

## TypeScript
[offical tutorial](https://www.typescriptlang.org/docs/home.html)
### Basic Type
- Boolean
`let isDone: boolean = false;`
- Number
`let decimal: number = 6;`
- String
`let color: string = "blue";`
- Array
`let list: number[] = [1, 2, 3];`
- Tuple
`let x: [string, number];`
`x = ["hello", 10];`
- Enum
`enum Color {Red, Green, Blue}`
`let c: Color = Color.Green;`
- Any
`let notSure`
- Void
`let unusable: void = undefined;`
Declaring variables of type `void` is not useful 
because you can only assign `null` or `undefined` to them
- Null & Undefine
subtypes of all other type
`let u: undefined = undefined;`
`let n: null = null;`
- Never
`never` is the return type for a function or one that never returns
- Object
```typescript
declare function create(o: object | null): void;

create({ prop: 0 });
create(null);
```
### loop
- forEach
```ts
var myArray = [1, 2, 3, 4];
myArray.name = "daocheng";

myArray.forEach(value => console.log(value));
// Can't break
```

- for in
```ts
for (var n in myArray) {
  console.log(n)
} // 1 2 3 4
```

- for of 
```ts
for (var n of myArray) {
  console.log(n)
} // 1 2 3 4
```
















