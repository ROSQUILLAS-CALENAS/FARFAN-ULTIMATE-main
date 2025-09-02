"""Audit trail module stub"""
class AuditTrail:
    def __init__(self):
        pass
    
    def log(self, event):
        pass

class MerkleNode:
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left  
        self.right = right
    
    def hash(self):
        return "dummy_hash"
