import discord
from discord.ext import commands
import time
import asyncio
from collections import defaultdict
from sqlalchemy.future import select

# Assuming database module contains these based on standard setups
try:
    from database import AsyncSessionLocal, GuildConfig
except ImportError:
    # Placeholder so the code can at least be parsed if imports are different
    AsyncSessionLocal = None
    GuildConfig = None

class AntiNuke(commands.Cog):
    """
    Anti-Nuke module that prevents users from mass-deleting channels, 
    roles, or mass-banning members.
    """
    def __init__(self, bot):
        self.bot = bot
        # Maps guild_id -> actor_id -> list of timestamps
        self.action_history = defaultdict(lambda: defaultdict(list))
        self.ACTION_LIMIT = 3
        self.TIME_WINDOW = 10  # seconds

    async def is_antinuke_enabled(self, guild_id: int) -> bool:
        if AsyncSessionLocal is None:
            return False
            
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(GuildConfig).where(GuildConfig.guild_id == guild_id)
            )
            config = result.scalar_one_or_none()
            return config.antinuke_enabled if config else False

    async def check_and_punish(self, guild: discord.Guild, actor: discord.Member, action_name: str):
        if actor.id == self.bot.user.id:
            return
            
        # Do not attempt to punish the guild owner as it will fail
        if actor.id == guild.owner_id:
            return
            
        now = time.time()
        history = self.action_history[guild.id][actor.id]
        history.append(now)
        
        # Prune old actions outside the time window
        history[:] = [t for t in history if now - t <= self.TIME_WINDOW]
        
        if len(history) > self.ACTION_LIMIT:
            # Trigger quarantine
            self.action_history[guild.id][actor.id].clear()
            
            try:
                # 1. Remove all roles (except @everyone)
                roles_to_remove = [role for role in actor.roles if role.name != "@everyone"]
                if roles_to_remove:
                    await actor.remove_roles(*roles_to_remove, reason="Anti-Nuke: Triggered destructive action threshold")
                
                # 2. Quarantine by Timing out (effectively sets interactive permissions to 0)
                # In discord.py, timeout removes send message/connect perms etc. for up to 28 days
                import datetime
                timeout_duration = datetime.timedelta(days=28)
                await actor.timeout(timeout_duration, reason="Anti-Nuke: Quarantine activated")
                
                # 3. Log to system channel or a dedicated log channel
                log_channel = guild.system_channel
                if log_channel:
                    embed = discord.Embed(
                        title="⚠️ ANTI-NUKE TRIGGERED ⚠️",
                        description=f"{actor.mention} (`{actor.id}`) performed more than {self.ACTION_LIMIT} destructive actions within {self.TIME_WINDOW} seconds.",
                        color=discord.Color.red()
                    )
                    embed.add_field(name="Action Taken", value="User had all roles removed and was placed in quarantine (timeout).")
                    await log_channel.send(embed=embed)
                    
            except discord.Forbidden:
                # The bot lacks permissions to punish this user (e.g., actor's top role > bot's top role)
                pass
            except discord.HTTPException:
                pass

    async def process_action(self, guild: discord.Guild, action_type: discord.AuditLogAction):
        if not await self.is_antinuke_enabled(guild.id):
            return
            
        # Give Discord's audit log a brief moment to update
        await asyncio.sleep(1.0)
        
        try:
            async for entry in guild.audit_logs(limit=1, action=action_type):
                # Ensure the entry is recent to prevent false positives from old logs
                if (discord.utils.utcnow() - entry.created_at).total_seconds() < 5:
                    actor = entry.user
                    if isinstance(actor, discord.Member):
                        await self.check_and_punish(guild, actor, action_type.name)
                break
        except discord.Forbidden:
            pass

    @commands.Cog.listener()
    async def on_guild_channel_delete(self, channel):
        if not isinstance(channel.guild, discord.Guild):
            return
        await self.process_action(channel.guild, discord.AuditLogAction.channel_delete)

    @commands.Cog.listener()
    async def on_member_ban(self, guild, user):
        await self.process_action(guild, discord.AuditLogAction.ban)

    @commands.Cog.listener()
    async def on_guild_role_delete(self, role):
        await self.process_action(role.guild, discord.AuditLogAction.role_delete)


async def setup(bot):
    await bot.add_cog(AntiNuke(bot))
