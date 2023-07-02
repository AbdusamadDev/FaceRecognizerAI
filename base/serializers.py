from rest_framework import serializers
from base.models import Employee


class EmployeeSerializer(serializers.ModelSerializer):
    folder = serializers.FileField(required=True)
    image = serializers.ImageField(required=True)

    class Meta:
        model = Employee
        fields = (
            "employee_id",
            "last_name",
            "first_name",
            "middle_name",
            "rank",
            "position",
            "image",
            "folder"
        )
