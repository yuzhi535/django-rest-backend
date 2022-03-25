# django-rest-backend

a  backend for my school homework

---

## 注册接口

### 必填：

* 用户名
* 密码
* 生日
* 身高
* 体重
* 手机号
* 姓名
* 头像
* 生日
* 身份证号

**以上使用form-data提交**

### 返回

* token（**暂定**）
* user_ID
* status： 200

### 若失败

* status： 404

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
