from rest_framework import serializers

from .models import CustomUser, User, Course


class CourseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Course
        fields = "__all__"


class CustomUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = "__all__"


class UserModelSerializer(serializers.ModelSerializer):
    phone_number = serializers.CharField(
        source='user.phone_number', required=True)
    password = serializers.CharField(source='user.password', style={'input_type': 'password'},
                                     required=True, write_only=True, max_length=256)
    gender = serializers.CharField(source='user.gender')

    class Meta:
        model = User
        # ['no_essential', 'name', 'avatar', 'height', 'weight', 'birthday', 'idcard_number', 'hobbies']
        fields = ['password', 'phone_number',
                  'gender', "name", 'height', 'weight']

        extra_kwargs = {
            "age": {
                "min_value": 0,
                "max_value": 120,
                "error_messages": {
                    "min_value": "年龄的最小值必须大于等于0",
                    "max_value": "年龄的最大值必须小于等于120",
                }
            },
            "height": {
                "min_value": 80,
                "max_value": 250,
                "error_messages": {
                    "min_value": "身高的最小值必须大于等于80",
                    "max_value": "身高的最大值必须小于等于250",
                }
            },
            "weight": {
                "min_value": 60,
                "error_messages": {
                    "min_value": "体重的最小值必须大于等于60",
                }
            },
        }

    def create(self, validated_data):
        newcustomuser = CustomUser.objects.create(gender=validated_data['user']['gender'],
                                                  password=validated_data['user']['password'],
                                                  phone_number=validated_data['user']['phone_number'], )
        newuser = User.objects.create(user=newcustomuser,
                                      name=validated_data['name'],
                                      #   avatar=validated_data['avatar'],
                                      height=validated_data['height'],
                                      weight=validated_data['weight'],
                                      #   birthday=validated_data['birthday'],
                                      #   idcard_number=validated_data['idcard_number'],
                                      #   hobbies=validated_data['hobbies']
                                      )

        return newuser

        # 添加多对多表中的记录
        # user.authors.add(*validated_data['authors'])
