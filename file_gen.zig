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
//// INCLUDE HEADERS ////////////////////////////

// used to find and measure externed declarations
const EXTERN_C: []const u8 = "extern \"C\"";

const EXTERN_HEADER_MACRO = 
    \\/* GENERATED FILE */
    \\
    \\#include "kernel_header.h"
    \\
    \\#if defined(__cplusplus)
    \\    #define EXTERN_C extern "C"
    \\#else
    \\    #define EXTERN_C extern
    \\#endif
    \\
    \\
;

const OVERLOAD_IMPORT: []const u8 = 
    \\
    \\const OverloadSet = @import("overloadset.zig").OverloadSet;
    \\
    \\const decls = @import("cimport.zig").C;
    \\
    \\
;

////////////////////////////////////////////////////
//// FILE GENERATOR IMPLEMENTATION /////////////////

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

system_arena: std.heap.ArenaAllocator,
system_allocator: std.mem.Allocator,
string_arena: std.heap.ArenaAllocator,
string_allocator: std.mem.Allocator,

source_abspaths: StringList,
target_abspaths: StringList,
source_filenames: StringList,
object_abspaths: StringList,

source_extension: []const u8,
source_directory: []const u8,
target_directory: []const u8,
zigsrc_directory: []const u8,
current_directory: []const u8,

pub fn init(config: FileGenConfig) *Self {

    const self: *Self = std.heap.page_allocator.create(Self) 
        catch @panic("Out of Memory");

    self.system_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);    
    self.system_allocator = self.system_arena.allocator();

    self.string_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);    
    self.string_allocator = self.string_arena.allocator();

    self.source_abspaths = StringList.initCapacity(self.system_allocator, 100)
        catch @panic("Failed to allocate source files capacity.");

    self.target_abspaths = StringList.initCapacity(self.system_allocator, 100)
        catch @panic("Failed to allocate targest files capacity.");

    self.object_abspaths = StringList.initCapacity(self.system_allocator, 100)
        catch @panic("Failed to allocate targest files capacity.");

    self.source_filenames = StringList.initCapacity(self.system_allocator, 100)
        catch @panic("Failed to allocate filenames capacity.");

    const cwd_path = std.fs.cwd().realpathAlloc(self.system_allocator, ".")
        catch @panic("Out of Memory");

    self.current_directory = cwd_path;

    self.source_directory = std.fs.path.join(self.system_allocator, &.{ cwd_path, config.source_directory })
        catch @panic("Out of Memory");

    self.target_directory = std.fs.path.join(self.system_allocator, &.{ cwd_path, config.target_directory })
        catch @panic("Out of Memory");

    self.zigsrc_directory = std.fs.path.join(self.system_allocator, &.{ cwd_path, config.zigsrc_directory })
        catch @panic("Out of Memory");

    self.source_extension = config.source_extension;

    return self;
}

pub fn deinit(self: *Self) void {
    self.system_arena.deinit();
    self.string_arena.deinit();
    std.heap.page_allocator.destroy(self);
}

pub fn appendSourceDirectory(self: *Self, source_name: []const u8) []const u8 {
    return std.fs.path.join(self.system_allocator, &.{ self.source_directory, source_name })
        catch @panic("Out of Memory");
}

pub fn appendLibraryDirectory(self: *Self, library_name: []const u8) []const u8 {
    return std.fs.path.join(self.system_allocator, &.{ self.zigsrc_directory, "lib", library_name })
        catch @panic("Out of Memory");
}

pub fn appendCudaDirectory(self: *Self, source_name: []const u8) []const u8 {
    return std.fs.path.join(self.system_allocator, &.{ self.zigsrc_directory, "cuda", source_name })
        catch @panic("Out of Memory");
}

pub fn appendExampleDirectory(self: *Self, example_name: []const u8) []const u8 {
    return std.fs.path.join(self.system_allocator, &.{ self.zigsrc_directory, "examples", example_name })
        catch @panic("Out of Memory");
}

pub fn appendTargetDirectory(self: *Self, target_name: []const u8) []const u8 {
    return std.fs.path.join(self.system_allocator, &.{ self.target_directory, target_name })
        catch @panic("Out of Memory");
}

pub fn appendZigsrcDirectory(self: *Self, zigsrc_name: []const u8) []const u8 {
    return std.fs.path.join(self.system_allocator, &.{ self.zigsrc_directory, zigsrc_name })
        catch @panic("Out of Memory");
}

fn joinSourceAbsPaths(self: *Self) []const u8 {
    return std.mem.join(self.system_allocator, " ", self.source_abspaths.items)
        catch @panic("Out of Memory");
}

fn joinTargetAbsPaths(self: *Self) []const u8 {
    return std.mem.join(self.system_allocator, " ", self.target_abspaths.items)
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
fn collectSources(self: *Self) !void {

    // lift off of stack
    const ObjectBuffer = struct {
        var data: [1024]u8 = undefined;
    };
    
    var dir: std.fs.Dir = try std.fs.openDirAbsolute(self.source_directory, .{ 
        .access_sub_paths = false, 
        .iterate = true, 
        .no_follow = true,
    });

    defer dir.close();
    
    var itr = dir.iterate();

    while (try itr.next()) |path| {
        
        const obj_name = replaceExtension(ObjectBuffer.data[0..], path.name, "o");
        const src = self.appendSourceDirectory(path.name);
        const trg = self.appendTargetDirectory(path.name);
        const obj = self.appendTargetDirectory(obj_name);

        try self.source_abspaths.append(src);
        try self.target_abspaths.append(trg);
        try self.object_abspaths.append(obj);
        try self.source_filenames.append(try self.system_allocator.dupe(u8, path.name));
    }
}


pub fn generate(self: *Self) void {

    if (replacer_sets.len == 0)
        @panic("No replacers specified");

    // this replacement algorithm works based on the structure
    // of the replacers up above. It will need to be done away
    // with if the replacement plan ever changes.
    std.log.info("Checking kernel source status...\n", .{});

    // iterate each source, replace the strings, then
    // save it to a target in the target directory
    self.collectSources() catch @panic("Failed to collect sources.");

    // determines generating declarations and overloads
    var modded_abspaths = std.ArrayListUnmanaged([]const u8){ };

    // buffer for generated kernel content
    var generations = std.ArrayListUnmanaged([]const u8){ };

    for (self.source_abspaths.items, self.target_abspaths.items) |src_path, trg_path| {

        if (!isModified(src_path, trg_path))
            continue;

        std.log.info("Generating:\n  {s}...\n", .{ src_path });

        modded_abspaths.append(self.system_allocator, trg_path)
            catch @panic("Failed to append to modded_targets");
        
        var gen_success: bool = false;
        
        const content = self.fileToString(src_path);

        for (MIN_LEVEL..MAX_LEVEL + 1) |cur_level| {

            // to enable multiple replacements per file, recycle
            // the last replacement to our current generation
            var current_gen: []const u8 = content;

            for (&replacer_sets) |*replacer_set| {
                for (replacer_set.replacers) |*replacer| {

                    if (replacer.level != cur_level)
                        continue;
                    
                    if (self.replace(current_gen, replacer_set.indicator, replacer.symbol)) |next_gen| {
                        current_gen = next_gen;
                        gen_success = true;
                    }
                }
            }

            if (!gen_success) @panic("Generation failed.");

            generations.append(self.system_allocator, current_gen)
                catch @panic("Failed to append generation.");
        }

        // join generations and write to file
        const file_content = std.mem.join(self.string_allocator, "\n", generations.items)
            catch @panic("Failed to join generations.");

        stringToFile(trg_path, file_content);

        // free our string capacity for next file set
        _ = self.string_arena.reset(.retain_capacity);

        generations.clearRetainingCapacity();    
    }    

    if (modded_abspaths.items.len != 0) {
        self.makeKernelDeclarations() 
            catch @panic("Failed to make kernel declarations.");

        self.makeKernelOverloads() 
            catch @panic("Failed to make overload sets.");
        
        std.log.info("Compiling kernel library...\n", .{});

        ScriptCompiler.compileStaticLibrary(.{
            .allocator = self.system_allocator,
            .modded_abspaths = modded_abspaths.items,
            .object_abspaths = self.object_abspaths.items,
            .current_directory = self.current_directory,
            .target_directory = self.target_directory,
            .lib_name = self.appendLibraryDirectory("mp_kernels.a")
        });
    }
}

fn makeKernelOverloads(self: *Self) !void {

    var overloadset_decls: []const u8 = "";
    var overloadset_args:  []const u8 = "";

    for (self.target_abspaths.items, self.source_filenames.items) |path, name| {

        const content = self.fileToString(path);

        const return_type: []const u8 = "void ";

        overloadset_args = "";

        var start: usize = 0;
        var last: usize = 0;

        while (true) {

            start = std.mem.indexOfPos(u8, content, last, "void launch_")
                orelse break;

            start += return_type.len;

            last = std.mem.indexOfPos(u8, content, start, "(")
                orelse @panic("Incomplete declaration.");

            overloadset_args = try std.mem.join(
                self.string_allocator, "", &.{ overloadset_args, "\tdecls.", content[start..last], ",\n" }
            );                
        }

        const name_stop = std.mem.indexOfScalar(u8, name, '.')
            orelse @panic("Target file does not have extension.");

        overloadset_decls = try std.mem.join(self.string_allocator, 
            "", &.{ overloadset_decls, "pub const ", name[0..name_stop], " = OverloadSet(.{\n", overloadset_args, "});\n\n"
        });
    }

    overloadset_decls = try std.mem.join(self.string_allocator, 
        "", &.{ OVERLOAD_IMPORT, overloadset_decls }
    );

    stringToFile(self.appendZigsrcDirectory("kernel_overloads.zig"), overloadset_decls);

    _ = self.string_arena.reset(.retain_capacity);
}


pub fn makeKernelDeclarations(self: *Self) !void {

    var declarations: []const u8 = EXTERN_HEADER_MACRO;

    for (self.target_abspaths.items) |trg_path| {    
        
        const trg_buffer = self.fileToString(trg_path);

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
            
            declarations = std.mem.join(self.string_allocator, "", &.{ declarations, "EXTERN_C", new_decl, ");\n" })
                catch @panic("Join Declaration: Out Of Memory");

            start = stop;
        }
        stringToFile(trg_path, trg_buffer);
    }

    stringToFile(self.appendCudaDirectory("kernel_decls.h"), declarations);

    _ = self.string_arena.reset(.retain_capacity);
}

pub fn makeCImport(self: *Self) void {

    const cimport_string = std.mem.join(self.string_allocator, "", &.{
        "pub const C = @cImport({\n",
            "\t@cInclude(\"", self.appendCudaDirectory("device_utils.h"), "\");\n",
            "\t@cInclude(\"", self.appendCudaDirectory("tensor_types.h"), "\");\n",
            "\t@cInclude(\"", self.appendCudaDirectory("kernel_decls.h"), "\");\n",
        "});\n"
    }) catch @panic("Failed to make cimport.zig");

    stringToFile(self.appendZigsrcDirectory("cimport.zig"), cimport_string);
}

fn fileToString(self: *Self, filename: []const u8) []u8 {

    const f = std.fs.openFileAbsolute(filename, .{}) catch @panic("Cannot open file.");
    defer f.close();
    
    const f_len = f.getEndPos() catch @panic("Could not get end position.");

    const string = self.string_allocator.alloc(u8, f_len) catch @panic("Out of memory.");

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

//fn appendTargetBuffer(
//    buffer: []u8,
//    start: usize,
//    tail: []const u8,
//) usize {
//
//    const end = std.mem.indexOfScalar(u8, tail, 0)
//        orelse tail.len;
//
//    if ((buffer.len - start) < end)
//        @panic("Buffer size too small");
//
//    const slice = buffer[start..start + end];
//
//    @memcpy(slice, tail[0..end]);
//
//    return start + end;
//}

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

pub fn replace(
    self: *Self,
    haystack: []const u8,
    needle: []const u8,
    replacement: []const u8
) ?[]const u8 {

    if (std.mem.indexOf(u8, haystack, needle) == null)
        return null;

    const needed = std.mem.replacementSize(
        u8, haystack, needle, replacement
    );

    const buffer = self.string_allocator.alloc(u8, needed) 
        catch @panic("Failed to allocate replacement buffer");

    const replacements = std.mem.replace(
        u8, haystack, needle, replacement, buffer
    );

    if (replacements == 0) 
        @panic("Failed to replace indicator in file.");

    return buffer; // indicate if we replaced something
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

