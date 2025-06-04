import java.util.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

public class Main {
    

    
    
    private static LinkedList[] adj;
    
    
    

    
    
    public static void main(String[] args) throws Exception {
        // your code goes here
        FastReader sc = new FastReader();
        int t = sc.nextInt();
        while (t-- > 0) {
            int n = sc.nextInt();
            Set<String> list1= new HashSet<>();
            Set<String> list2= new HashSet<>();
            Set<String> list3= new HashSet<>();
            int flag=0;
            for(int i=0;i<n;i++) {
                String s = sc.next();
                if (flag == 1) {
                   list1.add(s);
                }
                else
                {
                    if (s.length() == 1) {
                        flag = 1;
                    } else if (s.length() == 2) {
                        list2.add(s);
                        StringBuilder x = new StringBuilder();
                        x.append(s.charAt(1));
                        x.append(s.charAt(0));
                        if (list3.contains(x.toString()) || list2.contains(x.toString())) {
                            flag = 1;
                        }
                    } else {
                        list1.add(s);
                        list3.add(s.substring(0, 2));
                        StringBuilder x = new StringBuilder();
                        x.append(s.charAt(2));
                        x.append(s.charAt(1));
                        x.append(s.charAt(0));

                        if (list1.contains(x.toString())) flag = 1;
                        StringBuilder y = new StringBuilder();
                        y.append(s.charAt(2));
                        y.append(s.charAt(1));
                        if (list2.contains(y.toString())) flag = 1;

                    }


                }
            }

            if(flag!=1){
                System.out.println("NO");
            } else
                System.out.println("YES");
        }
    }

}
