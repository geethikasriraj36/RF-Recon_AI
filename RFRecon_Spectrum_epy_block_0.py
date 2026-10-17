"""
Embedded Python Blocks:
Each time this file is saved, GRC will instantiate the first class it finds.
"""

import numpy as np
from gnuradio import gr
import pmt

class blk(gr.basic_block):
    def __init__(self, Samp_Rate=2000000, Fft_Len=4096):
        """
        Arguments to this block will be configurable in the GRC GUI properties.
        Variable names must match the exact block keys: Samp_Rate and Fft_Len.
        """
        gr.basic_block.__init__(
            self,
            name='Frequency Extractor',   # Will appear in GRC
            in_sig=None,                  # No continuous stream inputs
            out_sig=None                  # No continuous stream outputs
        )
        self.samp_rate = Samp_Rate
        self.fft_len = Fft_Len

        # Register the input message port to receive data from Signal Detector
        self.message_port_register_in(pmt.intern('msg_in'))
        self.set_msg_handler(pmt.intern('msg_in'), self.handle_msg)

        # Register the output message port to send the final frequency to the GUI
        self.message_port_register_out(pmt.intern('out'))

    def handle_msg(self, msg):
        try:
            # 1. Parse GNU Radio PMT message layouts cleanly
            if pmt.is_pair(msg):
                meta = pmt.car(msg)
                payload = pmt.cdr(msg)
            else:
                payload = msg

            # 2. Extract numeric vector arrays safely
            if pmt.is_any_vector(payload):
                # Convert the PMT vector data into a standard Numpy processing array
                data = np.array(pmt.to_python(payload))
                
                if len(data) == 0:
                    return

                # 3. Locate the absolute peak energy index
                max_index = np.argmax(data)

                # 4. Perform centered FFT frequency calculations 
                if max_index >= (self.fft_len / 2):
                    bin_offset = max_index - self.fft_len
                else:
                    bin_offset = max_index

                # Formula mapping: Bin Step Size * Index Offset Location
                detected_freq = bin_offset * (self.samp_rate / self.fft_len)

                # 5. Publish a clean single Float message to stop display corruption
                self.message_port_pub(pmt.intern('out'), pmt.from_double(float(detected_freq)))

        except Exception as e:
            # Prevent silent failures by logging runtime exceptions to the terminal console
            print(f"[Frequency Extractor Error]: {e}")

    def work(self, input_items, output_items):
        # As an asynchronous event-driven block, it does not process stream blocks natively.
        return 0