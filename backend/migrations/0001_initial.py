# Generated by Django 4.0.2 on 2022-02-11 11:16

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Comment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('context', models.CharField(max_length=255, verbose_name='comment context')),
                ('likes', models.IntegerField(verbose_name='the like')),
            ],
        ),
        migrations.CreateModel(
            name='Course',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(choices=[('太极', 'taiji'), ('瑜伽', 'yujia')], max_length=255, verbose_name='course name')),
                ('rating', models.IntegerField(verbose_name='course rating')),
                ('assessment', models.FloatField(verbose_name='course assessment')),
                ('during_time', models.IntegerField(verbose_name='during time')),
            ],
        ),
        migrations.CreateModel(
            name='CustomUser',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('password', models.CharField(max_length=128, verbose_name='password')),
                ('last_login', models.DateTimeField(blank=True, null=True, verbose_name='last login')),
                ('phone_number', models.CharField(max_length=11, verbose_name='your phone number')),
                ('sex', models.CharField(choices=[('男', '男性'), ('女', '女性')], default='男', max_length=10)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('avatar', models.FileField(upload_to='', verbose_name='avatar')),
                ('height', models.IntegerField(verbose_name='your height')),
                ('weight', models.IntegerField(verbose_name='your weight')),
                ('birthday', models.DateField(verbose_name='your birthday')),
                ('hobbies', models.CharField(choices=[('太极', 'taiji'), ('瑜伽', 'yujia')], max_length=20, verbose_name='your hobbies')),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='backend.customuser', verbose_name='account')),
            ],
        ),
        migrations.CreateModel(
            name='User2Course',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('course', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='backend.course')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='backend.user')),
            ],
        ),
    ]
