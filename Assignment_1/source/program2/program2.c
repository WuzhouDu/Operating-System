#include <linux/err.h>
#include <linux/fs.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kmod.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/pid.h>
#include <linux/printk.h>
#include <linux/sched.h>
#include <linux/slab.h>

MODULE_LICENSE("GPL");

static struct task_struct *task;
struct wait_opts {
	enum pid_type wo_type;
	int wo_flags;
	struct pid *wo_pid;
	struct waitid_info *wo_info;
	int wo_stat;
	struct rusage *wo_rusage;
	wait_queue_entry_t child_wait;
	int notask_error;
};

extern pid_t kernel_clone(struct kernel_clone_args *args);

extern int do_execve(struct filename *filename,
		     const char __user *const __user *__argv,
		     const char __user *const __user *__envp);

extern long do_wait(struct wait_opts *wo);
extern struct filename *getname_kernel(const char *);

int my_execve(void)
{
	int status;
	const char path[] = "/tmp/test";
	struct filename *my_file = getname_kernel(path);

	// printk("[program2] : child process: do my_execve\n");
	printk("[program2] : child process\n");
	status = do_execve(my_file, NULL, NULL);

	// printk("status = %d\n", status);
	if (!status) {
		// printk("test program success!\n");
		return 0;
	}

	printk("not succeed in child process executing do_execve!\n");
	do_exit(status);
}

void my_wait(pid_t pid)
{
	// printk("begin my_wait\n");
	int a;
	int status;
	struct wait_opts wo;
	struct pid *wo_pid = NULL;
	enum pid_type type;
	type = PIDTYPE_PID;
	wo_pid = find_get_pid(pid);
	status = 0;

	wo.wo_type = type;
	wo.wo_pid = wo_pid;
	wo.wo_flags = WEXITED | WUNTRACED;
	wo.wo_info = NULL;
	wo.wo_stat = (int __user)status;
	wo.wo_rusage = NULL;

	a = do_wait(&wo);
	status = wo.wo_stat;
	// printk("do_wait return value is %d\n", a);
	if (status < 0b11111111) {
		status &= 0x7f;
		switch (status) {
		case 0:
			printk("[program2] : get SIGCHLD signal\n");
			// printk("[program2] : child process normal termination\n");
			break;

		case 1:
			printk("[program2] : get SIGHUP signal\n");
			// printk("[program2] : child process hung up\n");
			break;

		case 2:
			printk("[program2] : get SIGINT siganl\n");
			// printk("[program2] : child process interrupted\n");
			break;

		case 3:
			printk("[program2] : get SIGQUIT signal\n");
			// printk("[program2] : child process quit\n");
			break;

		case 4:
			printk("[program2] : get SIGILL signal\n");
			// printk("[program2] : child process illegal instruction\n");
			break;

		case 5:
			printk("[program2] : get SIGTRAP signal\n");
			// printk("[program2] : child process trapped\n");
			break;

		case 6:
			printk("[program2] : get SIGABRT signal\n");
			// printk("[program2] : child process abort\n");
			break;

		case 7:
			printk("[program2] : get SIGBUS signal\n");
			// printk("[program2] : child process bus error\n");
			break;

		case 8:
			printk("[program2] : get SIGFPE signal\n");
			// printk("[program2] : child process floating error\n");
			break;

		case 9:
			printk("[program2] : get SIGKILL signal\n");
			// printk("[program2] : child process get killed\n");
			break;

		case 11:
			printk("[program2] : get SIGSEGV signal\n");
			// printk("[program2] : child process segementation error\n");
			break;

		case 13:
			printk("[program2] : get SIGPIPE signal\n");
			// printk("[program2] : child process broken pipe\n");
			break;

		case 14:
			printk("[program2] : get SIGALRM signal\n");
			// printk("[program2] : child process get alarmed\n");
			break;

		case 15:
			printk("[program2] : get SIGTERM signal\n");
			// printk("[program2] : child process terminates\n");
			break;

		default:
			printk("[program2] : unkown signal\n");
		}
	} else {
		status = status >> 8;
		if (status == 19) {
			printk("[program2] : get SIGSTOP signal\n");
			// printk("[program2] : child process stopped\n");
		} else
			printk("[program2] : unkown signal\n");
	}

	printk("[program2] : child process terminated\n");

	printk("[program2] : The return signal is %d\n", status);

	put_pid(wo_pid);
	// printk("my_wait end\n");

	return;
}

// implement fork function
int my_fork(void *argc)
{
	pid_t pid;
	struct kernel_clone_args args = { .flags = SIGCHLD,
					  .stack = (unsigned long)&my_execve,
					  .stack_size = 0,
					  .parent_tid = NULL,
					  .child_tid = NULL,
					  .tls = 0,
					  .exit_signal = SIGCHLD };

	// set default sigaction for current process
	int i;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for (i = 0; i < _NSIG; i++) {
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}

	printk("[program2] : module_init kthread start\n");
	/* fork a process using kernel_clone or kernel_thread */
	pid = kernel_clone(&args);

	printk("[program2] : The child process has pid = %d\n", pid);
	printk("[program2] : This is the parent process, pid = %d\n",
	       (int)current->pid);

	/* execute a test program in child process */
	my_wait(pid);

	return 0;
}

static int __init program2_init(void)
{
	printk("[program2] : Module_init %s %d\n", "DuWuzhou", 120090575);

	/* write your code here */

	/* create a kernel thread to run my_fork */
	printk("[program2] : Module_init create kthread start\n");
	task = kthread_create(&my_fork, NULL, "forkProcess");

	if (!IS_ERR(task)) {
		// printk("[program2] : kernel tnread successfully created!!!!\n");
		wake_up_process(task);
	}

	return 0;
}

static void __exit program2_exit(void)
{
	printk("[program2] : Module_exit ./my\n");
}

module_init(program2_init);
module_exit(program2_exit);