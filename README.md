# django-rest-backend

a backend for my school homework

---

##
[**_部署教程_**](https://medium.com/faun/deploy-django-app-with-nginx-gunicorn-and-supervisor-on-ubuntu-server-ff58f5c201ac)

--- 

## 注册接口

### 必填：

* 用户名
* 密码
* 手机号
* 性别

**以上使用form-data提交**

### 返回

* token（**暂定**）
* 手机号
* 密码
* 性别
* 用户名
* 身高
* 体重
* 生日
* 爱好
* 身份证号
* 头像
* status： 200

### 若失败

* status：A400
* 'msg1': "注册信息有误,大概率是用户名未填"
* status：B400
* 'msg2': "两次密码输入不同"
* status：C400
* 'msg3': "该用户已注册"

## 登陆接口

### 必填

* 用户名
* 密码

**统一使用form-data形式**

### 返回

* token 临时token
* user_Id 用户ID
* status： 200

### 若失败

* status：A404
* 'msg1': "用户不存在"
* status：B404
* 'msg2': "密码输入错误"

## 上传视频接口

### 必填

* 用户ID
* 课程名
* 视频

### 返回

* status：204

## 分析视频接口

 ### 必填

* 用户token
* 文件名
* 课程

### 返回

* 状态： 是否成功接收视频

**form-data**

## 预测接口

### 必填

- 用户ID
- 课程名
- 文件名
- 动作内容(contents)，如白鹤亮翅之类的招式。字符串，招式为数字，后端会映射到相应的动作上，用空格分隔。如"1 2 3 4 5"表示 起式 野马分鬃 白鹤亮翅 搂膝拗步 手挥琵琶

### 返回

- status：204
- 预测完成的视频url(url)
- 评价(evaluate)
- 分数(score)
- 标准视频(criterion)

#### 若失败
- status:403
- msg:视频中没有人 或者 您距离摄像头太远了

**form-data**
