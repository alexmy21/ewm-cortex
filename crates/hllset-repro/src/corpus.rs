//! The shared reproduction corpus for the Phase 0/1 harnesses.

/// A small original corpus with repetitive structure — enough for a
/// character-level model to learn non-trivial next-token statistics.
pub const CORPUS: &str = "\
the cat sat on the mat and watched the rat.
the dog ran in the sun and chased the cat.
a fox hid in the box and saw the dog run.
the rat ran from the cat and sat by the log.
the dog sat by the log and watched the fox.
a cat and a dog sat on the mat in the sun.
the fox and the rat ran in the sun and hid.
the cat saw the rat and the dog saw the fox.
the dog ran to the box and the cat ran to the mat.
a rat hid in the box and a fox hid by the log.
the sun sat on the mat and the log sat in the box.
the cat and the fox ran from the dog and the rat.
a dog and a rat sat in the sun by the log.
the fox watched the cat and the rat watched the dog.
the rat ran in the box and the fox ran on the mat.
";
