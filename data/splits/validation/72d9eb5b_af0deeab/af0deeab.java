import java.io.*;
import java.util.*;

public class Solution {
    static int M = 998244353;
    static Random rng = new Random();

    private static boolean testCase(int n, int[] a, String colors) {
        boolean[] contains = new boolean[n];
        int any = 0, rem = n, idx;
        List<Integer> fromLeft = new ArrayList<>(), fromRight = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            if (colors.charAt(i) == 'B') {
                if (a[i] <= 0) {
                    return false;
                } else if (a[i] >= n) {
                    any++;
                } else {
                    fromLeft.add(a[i]);
                }
            } else {
                if (a[i] > n) {
                    return false;
                } else if (a[i] <= 1) {
                    any++;
                } else {
                    fromRight.add(a[i]);
                }
            }
        }

        sort(fromLeft);
        sort(fromRight);

        //System.out.println(fromLeft);
        //System.out.println(fromRight);

        idx = 1;

        for (int i = 0; i < fromLeft.size(); i++) {
            //try to assign a[i] to idx
            //System.out.println(String.format("Assign %d to %d", a[i], idx));
            if (fromLeft.get(i) < idx) {
                return false;
            } else {
                contains[idx - 1] = true;
                rem--;
                idx++;
            }
        }

        idx = n;

        for (int i = fromRight.size() - 1; i >= 0; i--) {
            //System.out.println(String.format("Assign %d to %d", a[i], idx));
            if (idx < fromRight.get(i)) {
                return false;
            } else {
                contains[idx - 1] = true;
                rem--;
                idx--;
            }
        }

        return any >= rem;
    }

    public static void main(String[] args) {
        FastScanner in = new FastScanner();
        PrintWriter out = new PrintWriter(System.out);
        int t = in.nextInt();  // Scanner has functions to read ints, longs, strings, chars, etc.
        //in.nextLine();
        for (int tt = 1; tt <= t; ++tt) {
            int n = in.nextInt();
            int[] a = in.readArray(n);
            String colors = in.next();

            out.println(testCase(n, a, colors) ? "YES" : "NO");
        }

        out.close();
    }

    

    private static void sort(int[] arr) {
        int temp, idx;

        for (int i = arr.length - 1; i > 0; i--) {
            idx = rng.nextInt(i + 1);

            temp = arr[i];
            arr[i] = arr[idx];
            arr[idx] = temp;
        }

        Arrays.sort(arr);
    }

    private static void sort(long[] arr) {
        long temp;
        int idx;

        for (int i = arr.length - 1; i > 0; i--) {
            idx = rng.nextInt(i + 1);

            temp = arr[i];
            arr[i] = arr[idx];
            arr[idx] = temp;
        }

        Arrays.sort(arr);
    }

    private static <T> void sort(T[] arr) {
        T temp;
        int idx;

        for (int i = arr.length - 1; i > 0; i--) {
            idx = rng.nextInt(i + 1);

            temp = arr[i];
            arr[i] = arr[idx];
            arr[idx] = temp;
        }

        Arrays.sort(arr);
    }

    private static <T> void sort(T[] arr, Comparator<? super T> cmp) {
        T temp;
        int idx;

        for (int i = arr.length - 1; i > 0; i--) {
            idx = rng.nextInt(i + 1);

            temp = arr[i];
            arr[i] = arr[idx];
            arr[idx] = temp;
        }

        Arrays.sort(arr, cmp);
    }

    private static <T extends Comparable<? super T>> void sort(List<T> list) {
        T temp;
        int idx;

        for (int i = list.size() - 1; i > 0; i--) {
            idx = rng.nextInt(i + 1);

            temp = list.get(i);
            list.set(i, list.get(idx));
            list.set(idx, temp);
        }

        Collections.sort(list);
    }

    private static <T> void sort(List<T> list, Comparator<? super T> cmp) {
        T temp;
        int idx;

        for (int i = list.size() - 1; i > 0; i--) {
            idx = rng.nextInt(i + 1);

            temp = list.get(i);
            list.set(i, list.get(idx));
            list.set(idx, temp);
        }

        Collections.sort(list, cmp);
    }
}
