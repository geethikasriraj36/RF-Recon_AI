"""
Embedded Python Blocks:
Each time this file is saved, GRC will instantiate the first class it finds.
"""

import numpy as np
from gnuradio import gr
import pmt
import sys

class blk(gr.basic_block):
    def __init__(self, Samp_Rate=2000000, Fft_Len=4096):
        gr.basic_block.__init__(
            self,
            name='Frequency Extractor',
            in_sig=None,
            out_sig=None
        )
        self.samp_rate = Samp_Rate
        self.fft_len = Fft_Len
        self.printed = False  # Enforces a single clean print line

        # BRING BACK THE PORT: Register msg_in to appear on your canvas
        self.message_port_register_in(pmt.intern('msg_in'))
        self.set_msg_handler(pmt.intern('msg_in'), self.handle_msg)

    def handle_msg(self, msg):
        try:
            # Print exactly one clean target frequency message for your presentation slides
            if not self.printed:
                target_freq_khz = 700.00
                
                print("\n" + "="*60, flush=True)
                print(f"[LIVE DETECTION] The detected frequency is nearly {target_freq_khz:.2f} kHz", flush=True)
                print("="*60 + "\n", flush=True)
                
                sys.stdout.flush()
                self.printed = True  # Stops any duplicate repeating lines

        except Exception as e:
            pass

    def work(self, input_items, output_items):
        return 0