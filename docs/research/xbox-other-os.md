# Running Linux or Windows on an Xbox Series X|S

Desk research and console measurements, 26 November 2026. No public route boots
Linux or Windows natively on a Series X or S; an operating system can only run
inside a Developer Mode app, under emulation.

## Why native boot is out

Series consoles only run Microsoft-signed boot code, enforced by Secure Boot and
a security co-processor. A Developer Mode app is a UWP package running as a
guest of the console's hypervisor and never reaches the boot chain.

The one public kernel exploit, [Collateral Damage](https://github.com/Coldblackice/xbox-collateral-damage)
(CVE-2024-30088, July 2024), gives kernel control of the SystemOS virtual
machine on kernel builds 10.0.25398.4478, .4908 and .4909 only. It was
[patched in .4910](https://xboxoneresearch.github.io/news/2024/08/15/news-recap.html)
and does not reach the hypervisor.

## What a Developer Mode app gets

Measured on a Series X by this fork (see [uwp-constraints.md](../uwp-constraints.md)
for the Series S figures and the full rules):

| Resource      | Value                                                            |
| ------------- | ---------------------------------------------------------------- |
| CPU           | 8 logical Zen 2 cores visible, 6 exclusive                       |
| Memory        | 5,120 MB as Game, 1,024 MB as App                                |
| GPU           | D3D12 compute; adapter "SraKmd arden"                            |
| Storage       | LocalState on `Q:`, 30,719 MB, shared by all sideloaded apps     |
| Network       | TCP sockets, including a LAN listener                            |
| Not available | Hardware virtualisation, kernel drivers, loading DLLs at runtime |

**JIT is unverified.** §4 of uwp-constraints.md records `VirtualAlloc` with
`PAGE_EXECUTE_*` as blocked. JIT-based emulators (Xenia, Dolphin) run in
Developer Mode, which suggests `VirtualAllocFromApp` / `VirtualProtectFromApp`
with the `codeGeneration` capability work. Nobody on this project has tested it,
and every fast emulator depends on it.

## Routes

| Route                                                             | Works today? | Effort        | Notes                                                                 |
| ----------------------------------------------------------------- | ------------ | ------------- | --------------------------------------------------------------------- |
| Stream a PC or VM to the console (Moonlight, RDP or VNC in Edge)  | Yes          | Hours         | Not on the console, but fully usable                                  |
| Linux in the Edge browser (v86, JSLinux)                          | Yes          | Minutes       | Slow; the research community lists "XBOX_Linux", a v86-emulated Linux |
| A small Linux emulator as a UWP app (for example TinyEMU, RISC-V) | Likely       | A weekend     | Disk image in LocalState; serial console or a web UI over the LAN     |
| QEMU (TCG) as a UWP app, x86-64 Linux                             | Possible     | Weeks         | Needs the JIT question answered                                       |
| Windows 10/11 in an emulator                                      | Very hard    | Weeks or more | 5 GB memory cap and no virtualisation                                 |
| Native Linux or dual boot                                         | No           | —             | No public route                                                       |

Suggested first step: a UWP Game app that writes `mov eax, 42; ret` into memory
from `VirtualAllocFromApp`, makes it executable with `VirtualProtectFromApp`,
and calls it.

## Sources

- [Collateral Damage repository](https://github.com/Coldblackice/xbox-collateral-damage)
- [Xbox One Research news recap, August 2024](https://xboxoneresearch.github.io/news/2024/08/15/news-recap.html)
- [Can you run Linux on an Xbox? (UMA Technology)](https://umatechnology.org/can-you-run-linux-on-an-xbox-the-answer-for-every-xbox-generation/)
- [What Xbox Developer Mode actually lets you do (BGR)](https://www.bgr.com/2272928/xbox-developer-mode-features-and-limitations/)
