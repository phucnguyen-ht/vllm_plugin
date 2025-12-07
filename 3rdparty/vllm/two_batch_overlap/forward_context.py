import threading

_forward_context_left = None
_forward_context_right = None

_left_tid = 0
_right_tid = 0

def init_tbo_forward_context(left_flag, tid):
    global _left_tid
    global _right_tid
    if left_flag:
        _left_tid = tid
    else:
        _right_tid = tid

def set_tbo_forward_context(_forward_context):
    global _forward_context_left
    global _forward_context_right
    tid = threading.get_ident()
    if tid == _left_tid:
        _forward_context_left = _forward_context
    else:
        _forward_context_right = _forward_context


def get_tbo_forward_context():
    tid = threading.get_ident()
    if tid == _left_tid:
        return _forward_context_left
    else:
        return _forward_context_right
