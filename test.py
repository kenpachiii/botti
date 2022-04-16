import asyncio

class Test:

    def __init__(self):

        self.event = asyncio.Event()

        self.attr_0 = 0
        self.attr_1 = 0
        self.attr_2 = 0

    async def func(self): 

        print(self.attr_0, self.attr_1, self.attr_2)

        # clear restarting event loop
        self.event.clear()

    def condition_0(self):
        print('checking condition 0')
        return False

    def condition_1(self):
        print('checking condition 1')
        return False

    async def producer_0(self):
        while True:
            await asyncio.sleep(1)
            self.attr_0 += 1

    async def producer_1(self):
        while True:
            await asyncio.sleep(1)
            self.attr_1 += 1

    async def producer_2(self):
        while True:
            await asyncio.sleep(5)
            self.attr_2 += 1

            # release after successful response
            self.event.set()

    async def consume(self):
        while True:

            if self.condition_0():
                await self.func()

            if self.condition_1():
                await self.func()
            
            # wait pending response else loop
            await self.event.wait()

test = Test()

loop = asyncio.get_event_loop()
loops = [test.producer_0(),test.producer_1(),test.producer_2(),test.consume()]
loop.run_until_complete(asyncio.gather(*loops))

