import java.io.DataInputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.TreeSet;
 
public class Main {
 
    private static void run() throws IOException {
        int n = in.nextInt();
        long[] b = new long[n];
        for (int i = 0; i < n; i++) {
            b[i] = in.nextInt();
        }
 
        TreeSet<Long> set = new TreeSet<>();
        set.add(b[0]);
        long last = b[0];
        long left = Long.MIN_VALUE;
        long right = Long.MAX_VALUE;
        boolean flag = true;
        for (int i = 1; i < n; i++) {
            if (b[i] > last) {
                set.add(right--);
                long pre_next = set.higher(last);
                if (pre_next < b[i]) {
                    flag = false;
                    break;
                } else if (pre_next == b[i]) {
                    set.add(right--);
                } else {
                    set.add(b[i]);
                }
            } else if (b[i] < last) {
                set.add(left++);
                long pre_next = set.lower(last);
                if (pre_next > b[i]) {
                    flag = false;
                    break;
                } else if (pre_next == b[i]) {
                    set.add(left++);
                } else {
                    set.add(b[i]);
                }
            } else {
                set.add(left++);
                set.add(right--);
            }
            last = b[i];
        }
 
        if (flag) {
            out.println("YES");
        } else {
            out.println("NO");
        }
    }
 
    public static void main(String[] args) throws IOException {
        in = new Reader();
        out = new PrintWriter(new OutputStreamWriter(System.out));
 
        int t = in.nextInt();
        for (int i = 0; i < t; i++) {
            run();
        }
 
        out.flush();
        in.close();
        out.close();
    }
 
    
 
    static final long mod = 1000000007;
 
    
 
    
 
    @SuppressWarnings("FieldCanBeLocal")
    private static Reader in;
    private static PrintWriter out;
 
    
 
    
 
    
 
    
 
    
}
