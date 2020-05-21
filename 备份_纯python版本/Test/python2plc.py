#!/usr/bin/env python
# -*- coding: utf_8 -*-


import serial

import modbus_tk
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu

PORT = 'COM3'





def main():

    logger = modbus_tk.utils.create_logger("console")

    try:

        master = modbus_rtu.RtuMaster(
            serial.Serial(port=PORT, baudrate=19200, bytesize=8, parity='E', stopbits=1, xonxoff=0)
        )
        master.set_timeout(5.0)#5s
        master.set_verbose(True)
        logger.info("connected")



        #send some queries
        #logger.info(master.execute(1, cst.READ_COILS, 0, 10))
        #logger.info(master.execute(1, cst.READ_DISCRETE_INPUTS, 0, 8))
        #logger.info(master.execute(1, cst.READ_INPUT_REGISTERS, 100, 3))
        #logger.info(master.execute(1, cst.READ_HOLDING_REGISTERS, 100, 12))
        #logger.info(master.execute(1, cst.WRITE_SINGLE_COIL, 7, output_value=1))

        logger.info(master.execute(1, cst.WRITE_SINGLE_REGISTER, 0x0FA6, output_value=3))  # 写入
        logger.info(master.execute(1, cst.WRITE_SINGLE_REGISTER, 0x0FA0, output_value=67)) #写入
        #logger.info(master.execute(1, cst.WRITE_MULTIPLE_COILS, 0, output_value=[1, 1, 0, 1, 1, 0, 1, 1]))
        #logger.info(master.execute(1, cst.WRITE_MULTIPLE_REGISTERS, 100, output_value=xrange(12)))

        logger.info(master.execute(1, cst.READ_HOLDING_REGISTERS, starting_address=0x0FA0, quantity_of_x=1))
        logger.info(master.execute(1, cst.READ_HOLDING_REGISTERS, starting_address=0x0FA6, quantity_of_x=1))  # 写入

    except modbus_tk.modbus.ModbusError as exc:
        logger.error("%s- Code=%d", exc, exc.get_exception_code())

if __name__ == "__main__":
    main()
