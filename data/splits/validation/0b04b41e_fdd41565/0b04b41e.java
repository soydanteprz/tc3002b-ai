import java.io.BufferedReader;
import java.io.*;
import java.util.*;

public class josph {

static BufferedReader br;


// Arrays.sort(time , (a1,a2) -> (a1[0]-a2[0])); 2d array sort lamda
public static void main(String[] args) throws Exception {
br = new BufferedReader(new InputStreamReader(System.in));
PrintWriter pr = new PrintWriter(System.out);
int tc = 1;
tc= cinI();
while(tc-->0){
int n =cinI();
String[] a= new String[n];

int[][] f =new int[10][n];

for(int i=0;i<n;i++){
a[i]=cin();
char[] x = a[i].toCharArray();

for(char c:x){
int index = c-'a';
f[index][i]+=1;
}
for(int j=0;j<10;j++){
int rem =x.length-f[j][i];
f[j][i]-=rem;
}
}
int max=0;
for(int j=0;j<10;j++){
Arrays.sort(f[j]);
int cnt=0;
int sum=0;
for(int i=n-1;i>=0;i--){
sum+=f[j][i];
if(sum>0){
cnt+=1;
}
else{
break;
}
}
max=Math.max(max,cnt);
}
System.out.println(max);
}

}



























public static String cin() throws Exception {
return br.readLine();
}









public static Integer cinI() throws Exception {
return Integer.parseInt(br.readLine());
}













}
