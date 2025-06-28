declare module 'gpu.js' {
    export interface IKernel<T extends any[], U> {
        (...args: T): U;
    }
    export class GPU {
        constructor(settings?: any);
        createKernel<T extends any[], U>(
            kernelFunction: (this: { thread: { x: number; y: number; z: number } }, ...args: T) => U,
            settings?: any
        ): IKernel<T, U>;
    }
}