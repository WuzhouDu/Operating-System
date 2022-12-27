#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

BUILD_SALT;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0x8db8d748, "module_layout" },
	{ 0x3c26a87b, "wake_up_process" },
	{ 0x641319cd, "kthread_create_on_node" },
	{ 0xff958bef, "kernel_clone" },
	{ 0x362e602b, "current_task" },
	{ 0x72332c9b, "put_pid" },
	{ 0xf37409c9, "do_wait" },
	{ 0x7243da1a, "find_get_pid" },
	{ 0xc959d152, "__stack_chk_fail" },
	{ 0x952664c5, "do_exit" },
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0x2d15d1b6, "do_execve" },
	{ 0xc5850110, "printk" },
	{ 0x85416d23, "getname_kernel" },
	{ 0xbdfb6dbb, "__fentry__" },
};

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "3CC2ED5B8F62DF685FB08BA");
