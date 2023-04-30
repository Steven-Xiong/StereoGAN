from .pwcnet import PWCNet_G, PWCNet_GC
from .loss import model_loss

__models__ = {
    "gwcnet-g": PWCNet_G,
    "gwcnet-gc": PWCNet_GC
}
