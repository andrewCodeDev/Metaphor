const std = @import("std");

const ScriptCompiler = @import("script_compiler.zig");

// level relates to the validity of a cast
// higher levels cannot result in lower levels
const Replacer = struct {
    symbol: []const u8,
    level: usize,
};

pub const ReplacerSet = struct {
    indicator: []const u8,
    replacers: []const Replacer,
};

const replacer_sets = [_]ReplacerSet {
    ReplacerSet { // real number replacers 
        .indicator = "RScalar",
        .replacers = &.{
            Replacer{ .symbol = "r16", .level = MIN_LEVEL + 0 },
            Replacer{ .symbol = "r32", .level = MIN_LEVEL + 1 },
            Replacer{ .symbol = "r64", .level = MIN_LEVEL + 2 },
        }
    },
    ReplacerSet { // real number replacers 
        .indicator = "CScalar",
        .replacers = &.{
            Replacer{ .symbol = "c16", .level = MIN_LEVEL + 0 },
            Replacer{ .symbol = "c32", .level = MIN_LEVEL + 1 },
            Replacer{ .symbol = "c64", .level = MIN_LEVEL + 2 },
        }
    },
    ReplacerSet { // real number replacers 
        .indicator = "RTensor",
        .replacers = &.{
            Replacer{ .symbol = "RTensor16", .level = MIN_LEVEL + 0 },
            Replacer{ .symbol = "RTensor32", .level = MIN_LEVEL + 1 },
            Replacer{ .symbol = "RTensor64", .level = MIN_LEVEL + 2 },
        }
    },
    ReplacerSet { // real number replacers 
        .indicator = "CTensor",
        .replacers = &.{
            Replacer{ .symbol = "CTensor16", .level = MIN_LEVEL + 0 },
            Replacer{ .symbol = "CTensor32", .level = MIN_LEVEL + 1 },
            Replacer{ .symbol = "CTensor64", .level = MIN_LEVEL + 2 },
        }
    },
};

const MIN_LEVEL: usize = 0;

const MAX_LEVEL: usize = blk: {
    var max_level: usize = 0;
    for (&replacer_sets) |*replacer_set| {
        for (replacer_set.replacers) |replacer| { max_level = @max(max_level, replacer.level); }
    } break :blk max_level;
};

const MAX_LEN: usize = blk: {
    var max_len: usize = 0;
    for (&replacer_sets) |*replacer_set| {
        for (replacer_set.replacers) |replacer| { max_len = @max(max_len, replacer.symbol.len); }
    } break :blk max_len;
};

/////////////////////////////////////////////////
//// DECLARATIONS ///////////////////////////////

// used to find and measure externed declarations
const EXTERN_C: []const u8 = "extern \"C\"";

const EXTERN_HEADER_MACRO = 
    \\/* GENERATED FILE */
    \\
    \\#include "../tensor_types.h"
    \\
    \\#if defined(__cplusplus)
    \\    #define EXTERN_C extern "C"
    \\#else
    \\    #define EXTERN_C extern
    \\#endif
    \\
    \\
;

const FileInfo = struct {
    path: []const u8,  
    modded: bool,
};

const ArrayList = std.ArrayList;
const StringList = ArrayList([]const u8);
const StringBuffer = [512]u8;
const FileList = ArrayList(FileInfo);

const FileGenConfig = struct {
    source_extension: []const u8,
    source_directory: []const u8,
    target_directory: []const u8,
    zigsrc_directory: []const u8,
};

const Self = @This();

arena: std.heap.ArenaAllocator,
allocator: std.mem.Allocator,
source_abspaths: StringList,
target_abspaths: StringList,
source_filenames: StringList,
source_extension: []const u8,
source_directory: []const u8,
target_directory: []const u8,
zigsrc_directory: []const u8,

pub fn init(config: FileGenConfig) *Self {

    const self: *Self = std.heap.page_allocator.create(Self) 
        catch @panic("Out of Memory");

    self.arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);    
    self.allocator = self.arena.allocator();

    self.source_abspaths = StringList.initCapacity(self.allocator, 100)
        catch @panic("Failed to allocate source files capacity.");

    self.target_abspaths = StringList.initCapacity(self.allocator, 100)
        catch @panic("Failed to allocate targest files capacity.");

    self.source_filenames = StringList.initCapacity(self.allocator, 100)
        catch @panic("Failed to allocate filenames capacity.");

    const cwd_path = std.fs.cwd().realpathAlloc(self.allocator, ".")
        catch @panic("Out of Memory");

    self.source_directory = std.fs.path.join(self.allocator, &.{ cwd_path, config.source_directory })
        catch @panic("Out of Memory");

    self.target_directory = std.fs.path.join(self.allocator, &.{ cwd_path, config.target_directory })
        catch @panic("Out of Memory");

    self.zigsrc_directory = std.fs.path.join(self.allocator, &.{ cwd_path, config.zigsrc_directory })
        catch @panic("Out of Memory");

    self.source_extension = config.source_extension;

    return self;
}

pub fn deinit(self: *Self) void {
    self.arena.deinit();
    std.heap.page_allocator.destroy(self);
}

pub fn appendSourceDirectory(self: *Self, source_name: []const u8) []const u8 {
    return std.fs.path.join(self.allocator, &.{ self.source_directory, source_name })
        catch @panic("Out of Memory");
}

pub fn appendTargetDirectory(self: *Self, target_name: []const u8) []const u8 {
    return std.fs.path.join(self.allocator, &.{ self.target_directory, target_name })
        catch @panic("Out of Memory");
}

pub fn appendZigsrcDirectory(self: *Self, zigsrc_name: []const u8) []const u8 {
    return std.fs.path.join(self.allocator, &.{ self.zigsrc_directory, zigsrc_name })
        catch @panic("Out of Memory");
}

pub fn isModified(source_path: []const u8, target_path: []const u8) bool {

    ////////////////////////////////////////////////////////////
    // open absolute paths as read-only to query their stats

    const source_file = std.fs.openFileAbsolute(source_path, .{})
        catch @panic("Failed to open source file");

    const target_file = std.fs.openFileAbsolute(target_path, .{})
        catch { return true; }; // file doesn't exist yet
    
    const source_stat = source_file.stat()
        catch @panic("Failed to open source file stat");

    const target_stat = target_file.stat()
        catch @panic("Failed to open source file stat");

    return target_stat.mtime < source_stat.mtime;
}

// we need to catch this error
fn collectSources(self: *Self) !bool {

    var modified: bool = false;

    var dir: std.fs.Dir = try std.fs.openDirAbsolute(self.source_directory, .{ 
        .access_sub_paths = false, 
        .iterate = true, 
        .no_follow = true,
    });

    defer dir.close();
    
    var itr = dir.iterate();

    while (try itr.next()) |path| {

        const src = self.appendSourceDirectory(path.name);
        const trg = self.appendTargetDirectory(path.name);


        if (isModified(src, trg)) {
            std.log.info("Modified: {s}", .{ src });
            modified = true;
        }

        try self.source_abspaths.append(src);
        try self.target_abspaths.append(trg);
        try self.source_filenames.append(try self.allocator.dupe(u8, path.name));
    }

    return modified;
}


pub fn generate(self: *Self) void {

    if (replacer_sets.len == 0)
        @panic("No replacers specified");

    // this replacement algorithm works based on the structure
    // of the replacers up above. It will need to be done away
    // with if the replacement plan ever changes.

    std.log.info("Checking kernel source status...\n", .{});

    var declarations: []const u8 = EXTERN_HEADER_MACRO;

    // iterate each source, replace the strings, then
    // save it to a target in the target directory
    const modified = self.collectSources() 
        catch @panic("Failed to collect sources");

    if (!modified) return; // we're all up to date

    std.log.info("Generating kernel source files...\n", .{});

    // get max replacement value to determine what the worst case
    // size needs to be for strings being replaced.

    for (self.source_abspaths.items, self.target_abspaths.items) |src_path, trg_path| {
        
        // give some size for the buffer to work

        const content = self.fileToString(src_path);

        // could be more sparing here, but it doesn't really matter.
        const repl_buffer: []u8 = self.allocator.alloc(u8, 2 * content.len)
            catch @panic("Failed to allocate the replace buffer");  

        defer self.allocator.free(repl_buffer);
        @memset(repl_buffer, 0);

        // could be more sparing here, but it doesn't really matter.
        const swap_buffer: []u8 = self.allocator.alloc(u8, 2 * content.len)
            catch @panic("Failed to allocate the swap buffer");  

        defer self.allocator.free(swap_buffer);
        @memset(swap_buffer, 0);

        // since sizes are always paired even across real and complex,
        // the number of generated functions is the same as a replacement
        // of a single numeric category
        const trg_buffer: []u8 = self.allocator.alloc(u8, 4 * repl_buffer.len)
            catch @panic("Failed to allocate the content buffer");  

        defer self.allocator.free(trg_buffer);
        @memset(trg_buffer, 0);

        var cur_level = MIN_LEVEL;
        var end_index: usize = 0;

        while (cur_level <= MAX_LEVEL) : (cur_level += 1) {
            
            @memcpy(swap_buffer[0..content.len], content);

            for (&replacer_sets) |*replacer_set| {
                for (replacer_set.replacers) |*replacer| {

                    if (replacer.level == cur_level)
                        if (replaceToBuffer(swap_buffer, replacer_set.indicator, replacer.symbol, repl_buffer))
                            @memcpy(swap_buffer, repl_buffer);
                }
            }
            end_index = appendTargetBuffer(trg_buffer, end_index, swap_buffer);
        }

        var start: usize = 0;

        while (true) { // break if we don't find another extern "C"

            start = std.mem.indexOfPos(u8, trg_buffer, start, EXTERN_C)
                orelse break;

            // get the end of the declaration
            const stop = std.mem.indexOfPos(u8, trg_buffer, start, ")")
                orelse @panic("Invalid declaration.");

            // we need to replace this with "EXTERN_C" for the macros to work
            start += EXTERN_C.len;

            const new_decl = trg_buffer[start..stop];
            
            declarations = std.mem.join(self.allocator, "", &.{ declarations, "EXTERN_C", new_decl, ");\n" })
                catch @panic("Join Declaration: Out Of Memory");

            start = stop;
        }

        stringToFile(trg_path, trg_buffer);
    }    
    stringToFile(self.appendTargetDirectory("kernel_decls.h"), declarations);

    self.makeKernelOverloads() 
        catch @panic("Failed to make overloads.");
        
    std.log.info("Compiling kernel library...\n", .{});

    ScriptCompiler.compileSharedLibrary(.{
        .allocator = self.allocator,
        .targets = self.target_abspaths.items,
        .libname  = self.appendZigsrcDirectory("libmp_kernels.so")
    });
}

fn makeKernelOverloads(self: *Self) !void {

    var overloadset_decls: []const u8 = "";
    var overloadset_args:  []const u8 = "";

    for (self.target_abspaths.items, self.source_filenames.items) |path, name| {

        const content = self.fileToString(path);

        overloadset_args = "";

        var start: usize = 42;
        var last: usize = 0;

        while (true) {

            start = std.mem.indexOfPos(u8, content, last, "launch_")
                orelse break;

             last = std.mem.indexOfPos(u8, content, start, "(")
                 orelse @panic("Incomplete declaration.");

             overloadset_args = try std.mem.join(
                 self.allocator, "", &.{ overloadset_args, "\tdecls.", content[start..last], ",\n" }
             );                
        }

        const name_stop = std.mem.indexOfScalar(u8, name, '.')
            orelse @panic("Target file does not have extension.");

        overloadset_decls = try std.mem.join(self.allocator, 
            "", &.{ overloadset_decls, "pub const ", name[0..name_stop], " = OverloadSet(.{\n", overloadset_args, "});\n\n"
        });
    }

    const import_head: []const u8 = 
        \\
        \\const OverloadSet = @import("overloadset.zig").OverloadSet;
        \\
        \\const decls = @import("cimport.zig").C;
        \\
        \\
    ;

    overloadset_decls = try std.mem.join(self.allocator, 
        "", &.{ import_head, overloadset_decls }
    );

    stringToFile(self.appendZigsrcDirectory("kernel_overloads.zig"), overloadset_decls);
}

fn joinSourceAbsPaths(self: *Self) []const u8 {
    return std.mem.join(self.allocator, " ", self.source_abspaths.items)
        catch @panic("Out of Memory");
}

fn joinTargetAbsPaths(self: *Self) []const u8 {
    return std.mem.join(self.allocator, " ", self.target_abspaths.items)
        catch @panic("Out of Memory");
}

pub fn reset(self: *Self) void {
    self.source_abspaths.deinit();
    self.target_abspaths.deinit();
    self.arena.reset(.retain_capacity);
}

fn fileToString(self: *Self, filename: []const u8) []u8 {

    const f = std.fs.openFileAbsolute(filename, .{}) catch @panic("Cannot open file.");
    defer f.close();
    
    const f_len = f.getEndPos() catch @panic("Could not get end position.");

    const string = self.allocator.alloc(u8, f_len) catch @panic("Out of memory.");

    _ = f.readAll(string) catch @panic("Could not read file.");

    return string;
}

fn stringToFile(path: []const u8, string: []const u8) void {

    const end = std.mem.indexOfScalar(u8, string, 0)
        orelse string.len;

    var file = std.fs.cwd().createFile(path, .{}) catch @panic("Failed to create file.");

    defer file.close();

    var writer = file.writer();

    _ = writer.writeAll(string[0..end]) catch @panic("Failed to write file.");
}

fn appendTargetBuffer(
    buffer: []u8,
    start: usize,
    tail: []const u8,
) usize {

    const end = std.mem.indexOfScalar(u8, tail, 0)
        orelse tail.len;

    if ((buffer.len - start) < end)
        @panic("Buffer size too small");

    const slice = buffer[start..start + end];

    @memcpy(slice, tail[0..end]);

    return start + end;
}

fn replaceExtension(
    buffer: []u8,
    inp_path: []const u8,
    out_ext: []const u8,
) []const u8 {
    if (inp_path.len == 0) {
        @panic("Empty path string.");
    }
    // incase there's weird characters,
    // we'll search string in reverse    
    var i = inp_path.len;

    while (i != 0) {

        i -= 1;

        if (inp_path[i] == '.')
            break;
    }

    if (i == 0) {
        @panic("Invalid path string.");
    }

    if (buffer.len < (i + out_ext.len)) {
        @panic("Buffer is too small.");
    }

    @memcpy(buffer[0..i + 1], inp_path[0..i + 1]);

    i += 1;

    @memcpy(buffer[i..i + out_ext.len], out_ext);

    const end = i + out_ext.len;

    return buffer[0..end];    
}

pub fn indicatorInSource(
    indicator: []const u8,
    source: []const u8,
) bool {
    if (std.mem.indexOf(u8, source, indicator)) |_| {
        return true;
    } else {
        return false;
    }
}

pub fn replaceToBuffer(
    haystack: []const u8,
    needle: []const u8,
    replacement: []const u8,
    buffer: []u8,
) bool {

    const end = std.mem.indexOfScalar(u8, haystack, 0)
        orelse haystack.len;

    // probably overkill, we calculate the buffer size before replacing
    const needed = std.mem.replacementSize(u8, haystack[0..end], needle, replacement);

    if (buffer.len < needed)
        @panic("Buffer size too small");

    const replacements = std.mem.replace(u8, haystack[0..end], needle, replacement, buffer);

    @memset(buffer[needed..], 0); // set new indicator for where stop

    return replacements != 0; // indicate if we replaced something
}

pub fn sameExtension(
    path: []const u8,
    extension: []const u8,
) bool {

    if (path.len < extension.len)
        return false;

    const begin = path.len - extension.len;

    return std.mem.eql(u8, path[begin..], extension);
}

