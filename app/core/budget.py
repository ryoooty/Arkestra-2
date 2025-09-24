"""
budget.py — управление токен-бюджетом:
- считает размер history+RAG+junior_meta
- режет низкоприоритетные куски если не помещается
- приоритет: history > RAG > junior_meta
"""
