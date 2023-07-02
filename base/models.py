from django.db import models


class Employee(models.Model):
    employee_id = models.CharField(max_length=50, unique=True)
    last_name = models.CharField(max_length=120)
    first_name = models.CharField(max_length=120)
    middle_name = models.CharField(max_length=120)
    rank = models.CharField(max_length=150, null=False)
    position = models.CharField(max_length=150, null=False)
    date_created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return "%s" % self.first_name
