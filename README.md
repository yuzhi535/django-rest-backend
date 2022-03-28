# django-rest-backend

a  backend for my school homework

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

* status：400
* 'msg1': "注册信息有误,大概率是用户名未填"
* 'msg2': "两次密码输入不同"
* 'msg3': "该用户已注册"
* status：404

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

* status： 404
* 'msg1': "用户不存在"
* 'msg2': "密码输入错误"

## 上传视频接口

### 必填

* 用户token
* 视频

### 返回

* user_ID
* token

## 分析视频接口

### 必填

* 用户token
* 文件名
* 课程

**form-data**
