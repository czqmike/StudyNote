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
     为网络层提供可靠的
  3. 网络层
  4. 传输层
  5. 应用层
























