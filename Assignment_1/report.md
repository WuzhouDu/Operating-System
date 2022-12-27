<h1 align="center"> CSC3150 Assignment1 Report </h1>

## Part 0: Student Information
Student ID: 120090575<br>Name: 杜五洲 DuWuzhou

## Part 1: Project Overview
The assignment includes two tasks.<br>
The task 1 mainly focuses on **manipulating multiple processes** in **user-mode** of operating system with **C POSIX library functions**. It mainly includes:

1. **Fork a child process** to **execute** 15 different test programs, each with a different signal raised.
2. Let the parent process **wait the child process** until it exits or stops, according to the **raised signal**.
3. After receiving the signal of child process, **print** the normal or abnormal termination information of child process.

The main flow chart of task 1 is:
![program1 flowchar](program1_flowchart.png)

The tasks 2 is similar to task 1, both cares about manipulating **multiple processes**. But task 2 instead focuses on the **kernel-mode** of operating system rather than in **user-mode**. What’s more, it requires **tracing the Linux kernel source code**, since in kernel-mode the only feasible APIs or functions are from there. This task mainly includes:

1. Create a **kernel thread** and run a self-implemented fork function in this thread to **fork a child process**. If the child process is successfully forked, print the parent process id and child process id.
2. After forking a child process, the parent process should **wait** the child process until it exits or stops with a self-implemented wait function. In the child process, it **executes** the test program and **receives signal** in the execution. This signal will be received by parent process.
3. After receiving the signal, print it according to the received value.

The main flow chart of task 2 is:
![program2 flowchar](program2_flowchart.png)

## Part 2: Code Detail Explanation

### program1.c:
![program1-1](program1-1.png)
Since program1 is in user-mode, so the code directly begins execution in main function. 
- As showed above, the variable `int status` is used to store the **return status of function `waitpid`**, which will be displayed in the following.
- The variable `pid_t pid` is the **process identifier** which is used to determine whether the execution is in the parent process or the child process. This variable is the return value of function `fork`.
- The function `fork` forks a child process to execute the program from the next line of code. It will return **-1** if **forking process fails**, else it will return the **process identifier** of the child process in the **parent process's following execution**, or it will return **0** in the **child process's following execution**.

Now we go deep into the `if` `else` statement blocks.
So, as stated above, if `pid` value is -1, it means that the `fork` function fails to fork a child process, the program will raise an error into the terminal.

![program1-2](program1-2.png)

If `pid` value is not -1, it means that the program successfully forks a child process and enters the following code blocks in **both two processes**.
So, to help program run smoothly in both processes, it is drafted in the style of separating two processes' execution with `if` condition. Now, `pid` is the identifier to distinguish the parent process and child process.

If `pid` value is 0, it means that execution now is the child process. Then the child process created an **argument variable** `arg` to store the `argv` variable **except the first argument** as the following executed file.

`execve` function executes the file whose path is `arg[0]` and giving argument to that executable file as `arg`, environment argument is just `NULL`.

If `pid` value is not 0, it means that execution goes on the parent process. So parent process needs to wait the child process through function `waitpid`. The function receives 3 arguments, first is the pid of child process, second is an integer pointer to store the **execution status** of the child process, third is the flag when the `waitpid` goes on. This program sets the flag to be `WUNTRACED` since the test program may raise `SIGSTOP` which stops the child process and does not raise exit signal. `WUNTRACED` means return to the parent process until child process stops or exits normally.

Now let's see what happens after receiving signal and `status`.
![program1-3](program1-3.png)

`WIFEXITED` macro queries the child termination status provided by the and `waitpid` function, and determines whether the child process **ended normally**. If child process ends normally, print the return status value.

`WIFSIGNALED` macro indicates whether the child process exits because of **raising a signal**. If true, consider the cases of all test signals and print respective information.

`WIFSTOPPED` macro indicates whether the child process **stops**. If true, print respective information.

### program2.c:
![program2-1](program2-1.png)
`insmod program2.ko` will enter `program2_init` function.
Firstly, create a **kernel thread** to begin execution of `my_fork`. `kthread_create` function receives three arguments which respectively represent the function will be executed in this thread, the data passed to the function, the process name. If no error detected, wake up this thread.

![program2-2](program2-2.png)

In `my_fork` function, the most significant part is about **forking a child process in function** `kernel_clone`. This function receive a unique arguments designed for this funcion, which is `struct kernel_clone_args`. 
`args.flags` is the set of flags of this newly process. 
`args.stack` is the beginning stack of user space, its size 
`args.size` is often 0. 
`args.parent_tid` and `args.child_tid` are both `NULL` since these two are the pids in the user space, which has nothing to do with this program in kernel mode. 
`args.exit_signal` is just the sent signal when exiting this process.

After calling `kernel_clone`, program gets the returned child process pid. This child pid is passed to execute `my_wait`.

![program2-3](program2-3.png)

`my_wait` function just receives one pid argument and passes the pid address to the `wait_opts` `wo_pid` attribute, indicating that the parent process should wait the process with this pid.
`wo_flags` indicates what signals the parent receives will trigger the following code execution after `do_wait`.
`do_wait` function stops the current process and let it wait until it receives the signal from the process with **pid** parameter. If successfully execution, it will return value 0 and **wo_stat** will be the **signal raised in the child process**.

Let's go deeper in the `status` handling.

![program2-4](program2-4.png)

If `status` is in the **lower eight bits**, it means that the **waited process signal exits with signal raised**. The lower eighth bit is to show whether core dump. So we can only consider the **lower seven bits** to get the signal-> `status &= 0x7f`

![program2-5](program2-5.png)

If `status` is larger than 0xff, it means the signal is `SIGSTOP` or `SIGTTIN`. **The lower eight bits of `status` are 0x7f, the second lower eight bits are the return signal**. So doing `status >>= 8` returns the signal value.

Now let's go to child process function `my_execve`.

![program2-6](program2-6.png)

This function is relatively easier. The address of executed file is in `path`. Pass `NULL` for both environment arguments and function arguments. The return status is stored in `status`.

## Part 3: How to Run My Code
1. **in program1 folder**:
```
vagrant@csc3150:~/csc3150/Assignment_1_120090575/source/program1$ make
vagrant@csc3150:~/csc3150/Assignment_1_120090575/source/program1$ ./program1 test
```

2. **in program2 folder**
```
vagrant@csc3150:~/csc3150/Assignment_1_120090575/source/program2$ make
vagrant@csc3150:~/csc3150/Assignment_1_120090575/source/program2$ sudo insmod program2.ko
vagrant@csc3150:~/csc3150/Assignment_1_120090575/source/program2$ sudo rmmod program2
vagrant@csc3150:~/csc3150/Assignment_1_120090575/source/program2$ dmesg -c
```

## Part 4: Environment and Compile Kernel
### Environment
Distributor ID: Ubuntu
Description:    Ubuntu 16.04.7 LTS
Release:        16.04
Codename:       xenial

kernel version: 5.10.146

### Compile Kernel
1. using `EXPORT_SYMBOL` macro to expose the functions in kernel source code, like `namei.c` so program can use out side the source code through `extern`. 
for example: `EXPORT_SYMBOL(kernel_clone);` after the implementation of `kernel_clone` function in `fork.c`, then `extern pid_t kernel_clone(struct kernel_clone_args *args);` in `program2.c`

2. using 
```
$make –j$(nproc)
$make modules_install
$make install
$reboot
```
to install the recompile the kernel.

## Part 5: What I've learned

1. all the kernel source code functions like `kernel_clone` `do_wait` `do_execve` and `kthread_created` etc.
Tracing the source code is such a challenging experience, but I've learnt a lot about how the **low-level systems** work.

2. The resources on Google are so abundant and I've got a lot hints from others' blog or answers to help me trace code.

## Part 6: Output Demo
### program1:
![](program1-demo.png)
![](program1-demo2.png)
![](program1-demo3.png)
![](program1-demo4.png)
![](program1-demo5.png)

### program2:
![](program2-demo1.png)
![](program2-demo2.png)
![](program2-demo3.png)
![](program2-demo4.png)
![](program2-demo5.png)
![](program2-demo6.png)
![](program2-demo7.png)
![](program2-demo8.png)
![](program2-demo9.png)
![](program2-demo10.png)
![](program2-demo11.png)
![](program2-demo12.png)
![](program2-demo13.png)
![](program2-demo14.png)
![](program2-demo15.png)