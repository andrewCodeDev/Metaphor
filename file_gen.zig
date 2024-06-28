const std = @import("std");

const ScriptCompiler = @import("script_compiler.zig");

const Fluent = @import("deps/fluent.zig");


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
    \\const decls = @import("cimport.zig").C;
    \\
    \\fn dispatch_array(tuple: anytype) [tuple.len]*const @TypeOf(tuple[0]) { return tuple; }
    \\
    \\
;

// replaces Scalar indicators
const REPLACERS: []const []const u8 = &.{ "r16", "r32", "r64" };

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
    const self: *Self = std.heap.page_allocator.create(Self) catch @panic("Out of Memory");

    self.system_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    self.system_allocator = self.system_arena.allocator();

    self.string_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    self.string_allocator = self.string_arena.allocator();

    self.source_abspaths = StringList.initCapacity(self.system_allocator, 100) catch @panic("Failed to allocate source files capacity.");
    self.target_abspaths = StringList.initCapacity(self.system_allocator, 100) catch @panic("Failed to allocate target files capacity.");
    self.object_abspaths = StringList.initCapacity(self.system_allocator, 100) catch @panic("Failed to allocate object files capacity.");
    self.source_filenames = StringList.initCapacity(self.system_allocator, 100) catch @panic("Failed to allocate filenames capacity.");

    const cwd_path = std.fs.cwd().realpathAlloc(self.system_allocator, ".") catch @panic("Out of Memory");

    self.current_directory = cwd_path;

    self.source_directory = std.fs.path.join(self.system_allocator, &.{ cwd_path, config.source_directory }) catch @panic("Out of Memory");
    self.target_directory = std.fs.path.join(self.system_allocator, &.{ cwd_path, config.target_directory }) catch @panic("Out of Memory");
    self.zigsrc_directory = std.fs.path.join(self.system_allocator, &.{ cwd_path, config.zigsrc_directory }) catch @panic("Out of Memory");
    self.source_extension = config.source_extension;

    return self;
}

pub fn deinit(self: *Self) void {
    self.system_arena.deinit();
    self.string_arena.deinit();
    std.heap.page_allocator.destroy(self);
}

pub fn append_source_directory(self: *Self, source_name: []const u8) []const u8 {
    return std.fs.path.join(self.system_allocator, &.{ self.source_directory, source_name }) catch @panic("Out of Memory");
}

pub fn append_library_directory(self: *Self, library_name: []const u8) []const u8 {
    return std.fs.path.join(self.system_allocator, &.{ self.zigsrc_directory, "lib", library_name }) catch @panic("Out of Memory");
}

// TODO: only cuda is supported currently - expand this in future releases
pub fn append_kernel_directory(self: *Self, source_name: []const u8) []const u8 {
    return std.fs.path.join(self.system_allocator, &.{ self.zigsrc_directory, "cuda", source_name }) catch @panic("Out of Memory");
}

pub fn append_example_directory(self: *Self, example_name: []const u8) []const u8 {
    return std.fs.path.join(self.system_allocator, &.{ self.zigsrc_directory, "examples", example_name }) catch @panic("Out of Memory");
}

pub fn append_target_directory(self: *Self, target_name: []const u8) []const u8 {
    return std.fs.path.join(self.system_allocator, &.{ self.target_directory, target_name }) catch @panic("Out of Memory");
}

pub fn append_core_directory(self: *Self, file_name: []const u8) []const u8 {
    return std.fs.path.join(self.system_allocator, &.{ self.zigsrc_directory, "core", file_name }) catch @panic("Out of Memory");
}

pub fn append_algo_directory(self: *Self, file_name: []const u8) []const u8 {
    return std.fs.path.join(self.system_allocator, &.{ self.zigsrc_directory, "algo", file_name }) catch @panic("Out of Memory");
}

pub fn append_ops_directory(self: *Self, file_name: []const u8) []const u8 {
    return std.fs.path.join(self.system_allocator, &.{ self.zigsrc_directory, "ops", file_name }) catch @panic("Out of Memory");
}

fn join_source_absoute_paths(self: *Self) []const u8 {
    return std.mem.join(self.system_allocator, " ", self.source_abspaths.items) catch @panic("Out of Memory");
}

fn join_target_absolute_paths(self: *Self) []const u8 {
    return std.mem.join(self.system_allocator, " ", self.target_abspaths.items) catch @panic("Out of Memory");
}

pub fn is_modified(source_path: []const u8, target_path: []const u8) bool {

    ////////////////////////////////////////////////////////////
    // open absolute paths as read-only to query their stats

    const source_file = std.fs.openFileAbsolute(source_path, .{}) catch @panic("Failed to open source file");

    const target_file = std.fs.openFileAbsolute(target_path, .{}) catch {
        return true;
    }; // file doesn't exist yet

    const source_stat = source_file.stat() catch @panic("Failed to open source file stat");

    const target_stat = target_file.stat() catch @panic("Failed to open source file stat");

    return target_stat.mtime < source_stat.mtime;
}

// we need to catch this error
fn collect_sources(self: *Self) !void {

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
        const obj_name = replace_extension(ObjectBuffer.data[0..], path.name, "o");
        const src = self.append_source_directory(path.name);
        const trg = self.append_target_directory(path.name);
        const obj = self.append_target_directory(obj_name);

        try self.source_abspaths.append(src);
        try self.target_abspaths.append(trg);
        try self.object_abspaths.append(obj);
        try self.source_filenames.append(try self.system_allocator.dupe(u8, path.name));
    }
}

pub fn make_device_kernels(self: *Self) void {

    // this replacement algorithm works based on the structure
    // of the replacers up above. It will need to be done away
    // with if the replacement plan ever changes.
    std.log.info("Checking kernel source status...\n", .{});

    // iterate each source, replace the strings, then
    // save it to a target in the target directory
    self.collect_sources() catch @panic("Failed to collect sources.");

    // determines generating declarations and overloads
    var modded_abspaths = std.ArrayListUnmanaged([]const u8){};

    // buffer for generated kernel content
    var generations = std.ArrayListUnmanaged([]const u8){};

    for (self.source_abspaths.items, self.target_abspaths.items) |src_path, trg_path| {
        
        if (!is_modified(src_path, trg_path))
            continue;

        std.log.info("Generating:\n  {s}...\n", .{src_path});

        modded_abspaths.append(self.system_allocator, trg_path) catch @panic("Failed to append to modded_targets");

        const content = self.file_to_string(src_path);

        for (REPLACERS) |replacer| {
            const new_content = self.string_allocator.dupe(u8, content) catch @panic("Failed to duplicate content");
            const generated = Fluent.init(new_content).replace(.regex, "Scalar", replacer).items;
            generations.append(self.system_allocator, generated) catch @panic("Faioled to append generation");
        }

        // join generations and write to file
        const file_content = std.mem.join(self.string_allocator, "\n", generations.items) 
            catch @panic("Failed to join generations.");

        string_to_file(trg_path, file_content);

        // free our string capacity for next file set
        _ = self.string_arena.reset(.retain_capacity);

        generations.clearRetainingCapacity();
    }

    if (modded_abspaths.items.len != 0) {
        self.make_kernel_declarations() catch @panic("Failed to make kernel declarations.");

        self.make_kernel_overloads() catch @panic("Failed to make overload sets.");

        std.log.info("Compiling kernel library...\n", .{});

        ScriptCompiler.compile_static_library(.{ 
            .allocator = self.system_allocator,
            .modded_abspaths = modded_abspaths.items,
            .object_abspaths = self.object_abspaths.items, 
            .current_directory = self.current_directory, 
            .target_directory = self.target_directory, 
            .lib_name = self.append_library_directory("mp_kernels.a") 
        });
    }
}

fn make_kernel_declarations(self: *Self) !void {
    var declarations: []const u8 = EXTERN_HEADER_MACRO;

    for (self.target_abspaths.items) |trg_path| {
        const content = self.file_to_string(trg_path);

        var itr = Fluent.match("extern \"C\" void \\w+\\s?\\([^)]+\\)", content);

        while (itr.next()) |decl| { // break if we don't find another extern "C"
            declarations = std.mem.join(self.string_allocator, "", &.{
                declarations,
                "EXTERN_C",
                decl.trim(.left, .regex, "extern \"C\"").items,
                ";\n",
            }) catch @panic("Join Declaration: Out Of Memory");
        }
    
        string_to_file(trg_path, content);
    }

    // TODO: only cuda is supported currently - expand this in future releases
    string_to_file(self.append_kernel_directory("kernel_decls.h"), declarations);

    _ = self.string_arena.reset(.retain_capacity);
}

pub fn make_device_utils(
    self: *Self, 
    src_path: []const u8,
    trg_path: []const u8,
) void {
    ScriptCompiler.compile_shared_file(self.system_allocator, src_path, trg_path);
}

fn make_kernel_overloads(self: *Self) !void {
    var overloadset_decls: []const u8 = "";
    var overloadset_args: []const u8 = "";

    for (self.target_abspaths.items, self.source_filenames.items) |path, name| {
        const content = self.file_to_string(path);
        overloadset_args = "";

        var itr = Fluent.match("void launch_\\w+", content);

        while (itr.next()) |decl| {

            overloadset_args = try std.mem.join(self.string_allocator, "", &.{
                overloadset_args, "\tdecls.",  decl.trim(.left, .regex, "void ").items, ",\n" 
            });
        }

        const name_stop = Fluent.init(name).find(.scalar, '.') orelse @panic("Target file does not have extension.");

        overloadset_decls = try std.mem.join(self.string_allocator, "", &.{
            overloadset_decls, "pub const ", name[0..name_stop], " = dispatch_array(.{\n", overloadset_args, "});\n\n"
        });
    }

    overloadset_decls = try std.mem.join(self.string_allocator, "", &.{ OVERLOAD_IMPORT, overloadset_decls });

    // create main kernel dispatching file exported through core.root
    string_to_file(self.append_core_directory("kernels.zig"), overloadset_decls);

    _ = self.string_arena.reset(.retain_capacity);
}

pub fn make_core_cimport(self: *Self) void {
    const cimport_string = std.mem.join(self.string_allocator, "", &.{
        "pub const C = @cImport({\n",
            "\t@cInclude(\"", self.append_kernel_directory("device_utils.h"), "\");\n",
            "\t@cInclude(\"", self.append_kernel_directory("tensor_types.h"), "\");\n",
            "\t@cInclude(\"", self.append_kernel_directory("kernel_decls.h"), "\");\n", 
        "});\n",
    }) catch @panic("Failed to make cimport.zig");

    string_to_file(self.append_core_directory("cimport.zig"), cimport_string);
}

fn file_to_string(self: *Self, filename: []const u8) []u8 {
    const f = std.fs.openFileAbsolute(filename, .{}) catch @panic("Cannot open file.");
    defer f.close();

    const f_len = f.getEndPos() catch @panic("Could not get end position.");

    const string = self.string_allocator.alloc(u8, f_len) catch @panic("Out of memory.");

    _ = f.readAll(string) catch @panic("Could not read file.");

    return string;
}

pub fn reset(self: *Self) void {
    _ = self.system_arena.reset(.retain_capacity);
    _ = self.string_arena.reset(.retain_capacity);
    self.source_abspaths.resize(0) catch unreachable;
    self.target_abspaths.resize(0) catch unreachable;
    self.object_abspaths.resize(0) catch unreachable;
    self.source_filenames.resize(0) catch unreachable;   
}

fn string_to_file(path: []const u8, string: []const u8) void {
    const end = std.mem.indexOfScalar(u8, string, 0) orelse string.len;

    var file = std.fs.cwd().createFile(path, .{}) catch @panic("Failed to create file.");
    defer file.close();

    var writer = file.writer();
    _ = writer.writeAll(string[0..end]) catch @panic("Failed to write file.");
}

fn replace_extension(
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

    @memcpy(buffer[0 .. i + 1], inp_path[0 .. i + 1]);

    i += 1;

    @memcpy(buffer[i .. i + out_ext.len], out_ext);

    const end = i + out_ext.len;

    return buffer[0..end];
}

