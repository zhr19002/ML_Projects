import lightning.pytorch as pl
from config import *
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

class EpochLogger(Callback):
    def on_validation_epoch_end(self, trainer, _):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch + 1
        train_loss = trainer.callback_metrics.get('train_loss')
        val_loss = trainer.callback_metrics.get('val_loss')
        train_loss = float(train_loss) if train_loss is not None else float('nan')
        val_loss = float(val_loss) if val_loss is not None else float('nan')
        lr = trainer.optimizers[0].param_groups[0]['lr']
        print(f'Epoch {epoch:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={lr:.2e}')

def build_trainer(filename, max_epochs, patience):
    trainer = pl.Trainer(
        accelerator = 'gpu',
        devices = 1,
        max_epochs = max_epochs,
        gradient_clip_val = 0.1,
        callbacks = [
            EpochLogger(),
            EarlyStopping(monitor='val_loss', patience=patience),
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(dirpath=CKPT_DIR, filename=filename, monitor='val_loss', mode='min', save_top_k=1),
        ],
        enable_progress_bar = False,
        log_every_n_steps = 1e6,
        logger = TensorBoardLogger(LOG_DIR, name=filename),
    )
    return trainer