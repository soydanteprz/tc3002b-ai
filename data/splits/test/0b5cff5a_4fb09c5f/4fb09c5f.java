import java.io.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;
import java.util.stream.Collectors;

public class Trial {
    public static void main(String[] args) throws IOException {

        Scanner sc = new Scanner(System.in);
        PrintWriter pw = new PrintWriter(System.out);

        int q = sc.nextInt();
        while (q-- > 0) {
            int n = sc.nextInt();
            int k = sc.nextInt();
            int[] arr = new int[k];
            int[] t = new int[k];
            HashMap<Integer, Integer> hm = new HashMap<>();
            for (int i = 0; i < k; i++) {
                arr[i] = sc.nextInt() - 1;
            }

            for (int i = 0; i < k; i++) {
                t[i] = sc.nextInt();
                hm.put(arr[i], t[i]);
            }

            int[] left = new int[n];
            int[] right = new int[n];
            left[0] = hm.getOrDefault(0, -1);
            right[n - 1] = hm.getOrDefault(n - 1, -1);
            for (int i = 1; i < n; i++) {
                if (hm.containsKey(i)) {
                    if (left[i - 1] < 0) {
                        left[i] = hm.get(i);
                    } else {
                        left[i] = Math.min(hm.get(i), left[i - 1] + 1);
                    }
                } else {
                    left[i] = left[i - 1] < 0 ? -1 : left[i - 1] + 1;
                }
            }
            for (int i = n - 2; i >= 0; i--) {
                if (hm.containsKey(i)) {
                    if (right[i + 1] < 0) {
                        right[i] = hm.get(i);
                    } else {
                        right[i] = Math.min(hm.get(i), right[i + 1] + 1);
                    }
                } else {
                    right[i] = right[i + 1] < 0 ? -1 : right[i + 1] + 1;
                }
            }
            for (int i = 0; i < n; i++) {
                if (left[i] < 0) {
                    pw.print(right[i] + " ");
                } else if (right[i] < 0) {
                    pw.print(left[i] + " ");
                } else {
                    pw.print(Math.min(left[i], right[i]) + " ");
                }
            }
            pw.println();
        }
        pw.flush();
        pw.close();
    }

    // inclusive
    

    

    
}
