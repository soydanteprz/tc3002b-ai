import java.io.*;
import java.util.*;

import static java.lang.Math.*;

//Think through the entire logic before jump into coding!
//If you are out of ideas, take a guess! It is better than doing nothing!
//Read both C and D, it is possible that D is easier than C for you!

//Be aware of integer overflow!
//If you find an answer and want to return immediately, don't forget to flush before return!

public class C {
    static InputReader in;
    static PrintWriter out;

    public static void main(String[] args) {
        //initReaderPrinter(true);
        initReaderPrinter(false);
        solve(in.nextInt());
        //solve(1);        
    }

    static void solve(int testCnt) {
        for (int testNumber = 0; testNumber < testCnt; testNumber++) {
            int n = in.nextInt();
            char[] a = in.next().toCharArray(), b = in.next().toCharArray();
            int match0 = 0, match1 = 0, mismatch10 = 0, mismatch01 = 0;
            for(int i = 0; i < n; i++) {
                if(a[i] == b[i]) {
                    if(a[i] == '0') match0++;
                    else match1++;
                }
                else {
                    if(a[i] == '0') mismatch01++;
                    else mismatch10++;
                }
            }
            if(mismatch01 + mismatch10 == 0) out.println(0);
            else {
                if(match1 - match0 == 1 && mismatch01 == mismatch10) {
                    out.println(min(match0 + match1, mismatch01 + mismatch10));
                }
                else if(match1 - match0 == 1) {
                    out.println(match0 + match1);
                }
                else if(mismatch01 == mismatch10) {
                    out.println(mismatch01 + mismatch10);
                }
                else out.println(-1);
            }
        }
        out.close();
    }

    static void initReaderPrinter(boolean test) {
        if (test) {
            try {
                in = new InputReader(new FileInputStream("src/input.in"));
                out = new PrintWriter(new FileOutputStream("src/output.out"));
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            in = new InputReader(System.in);
            out = new PrintWriter(System.out);
        }
    }

    
}
