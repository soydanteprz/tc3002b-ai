import java.io.*;
import java.util.*;

public class Main {

    public static void main(String[] args) {
        MyScanner scan = new MyScanner();
        out = new PrintWriter(new BufferedOutputStream(System.out));

        int t = scan.nextInt();
        for (int i = 0; i < t; i++) {
            int num = scan.nextInt();
            int[] sequence = new int[num];
            for (int j = 0; j < num; j++) {
                sequence[j] = scan.nextInt();
            }

            int[] setSum = new int[(int) Math.pow(3, num)];

            for (int k = 0; k < num; k++) {
                int pot =(int)Math.pow(3, k);
                
                for (int j = 1; j < setSum.length; j++) {
                    if(j % (pot*3) >= pot ){
                        if(j % (pot*3) >= pot){
                            if(j % (pot*3) >= pot*2){
                                setSum[j] += sequence[k];
                            }else{
                                setSum[j] -= sequence[k];
                            }
                        }

                    }
                }
            }
            String output = "NO";
            for (int j = 1; j < setSum.length; j++) {
                if(setSum[j] == 0){
                    output = "YES";
                    break;
                }
            }
            out.println(output);
        }
        out.close();
    }

    public static PrintWriter out;
    
}
