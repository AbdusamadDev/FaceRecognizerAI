# Generated by Django 4.2.2 on 2023-07-02 19:38

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Employee',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('employee_id', models.CharField(max_length=50)),
                ('last_name', models.CharField(max_length=120)),
                ('first_name', models.CharField(max_length=120)),
                ('middle_name', models.CharField(max_length=120)),
                ('rank', models.CharField(max_length=150)),
                ('position', models.CharField(max_length=150)),
                ('image', models.ImageField(upload_to='id/')),
                ('date_created', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
