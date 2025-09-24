class ValidationError(Exception):
    pass


def validate(instance, schema):
    return True


__all__ = ["validate", "ValidationError"]
